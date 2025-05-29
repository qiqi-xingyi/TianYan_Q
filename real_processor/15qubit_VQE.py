# --*-- conding:utf-8 --*--
# @time:5/29/25 14:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:15qubit_VQE.py

from mindquantum.core import Circuit, apply, RY, CNOT
from cqlib import TianYanPlatform, QuantumLanguage
from copy import deepcopy
import json
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.common.parameter import Parameter
import logging
import os

# Configure logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('log/15qubit_vqe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_cost(result: dict):
    return json.loads(result['probability'])['0']

def hardware_run(circuits: list[Circuit], shots: int = 1000):
    tasks = [circ.to_qcis(parametric=False) for circ in circuits]
    query_ids = platform.submit_experiment(
        circuit=tasks,
        language=QuantumLanguage.QCIS,
        num_shots=shots
    )
    results = platform.query_experiment(
        query_id=query_ids,
        max_wait_time=600,
        sleep_time=5
    )
    return [compute_cost(r) for r in results]

def get_cost_with_grads(circuit: Circuit):
    encoder_names = circuit.encoder_params_name
    ansatz_names = circuit.ansatz_params_name
    shift = np.pi / 2

    def grad_ops(encoder_values, ansatz_values):
        param_dict_e = dict(zip(encoder_names, encoder_values))
        base_circuit = circuit.apply_value(param_dict_e)

        circuits = []
        circuits.append(base_circuit.apply_value(dict(zip(ansatz_names, ansatz_values))))

        circuits_p = []
        circuits_n = []
        for i in range(len(ansatz_values)):
            v_p = ansatz_values.copy()
            v_n = ansatz_values.copy()
            v_p[i] += shift
            v_n[i] -= shift
            circuits_p.append(base_circuit.apply_value(dict(zip(ansatz_names, v_p))))
            circuits_n.append(base_circuit.apply_value(dict(zip(ansatz_names, v_n))))

        circuits.extend(circuits_p)
        circuits.extend(circuits_n)

        costs = hardware_run(circuits)
        cost = costs[0]
        cost_p = ms.Tensor(costs[1:1+len(ansatz_values)])
        cost_n = ms.Tensor(costs[1+len(ansatz_values):])
        grads = (cost_p - cost_n) / 2
        return cost, grads

    return grad_ops

if __name__ == '__main__':
    # Create 15-qubit circuit
    num_qubits = 15
    circ = Circuit(num_qubits=num_qubits)

    # Encoder layer
    for i in range(num_qubits):
        circ += RY(f'alpha{i}').on(i)
    circ.as_encoder()

    # Ansatz layer with chain entanglement
    for i in range(num_qubits):
        circ += RY(f'theta{i}').on(i)
    for i in range(num_qubits - 1):
        circ += CNOT().on(i, i + 1)
    circ.measure_all()

    # Map to physical qubits
    physical_qubits = [259,273,287,301,315,329,343,357,371,385,399,413,427,441,455]
    circ = apply(circ, physical_qubits)

    # Get gradient operator
    grad_ops = get_cost_with_grads(circ)

    # Initialize encoder values and parameters
    encoder_values = ms.Tensor([np.pi / 2] * num_qubits)
    thetas = [Parameter(ms.Tensor([0.0]), name=f'theta{i}') for i in range(num_qubits)]
    optimizer = nn.SGD(thetas, learning_rate=0.5)

    # Connect to quantum platform
    login_key = '/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA='
    machine_name = 'tianyan504'
    platform = TianYanPlatform(login_key=login_key, machine_name=machine_name)

    # Training loop
    for step in range(20):
        ansatz_values = ms.Tensor([t.asnumpy()[0] for t in thetas])
        cost, grads = grad_ops(encoder_values, ansatz_values)
        grads_list = grads.asnumpy().tolist()
        thetas_list = [float(t.asnumpy()[0]) for t in thetas]

        logger.info(f'Step {step:02d} | Thetas: {thetas_list} | Cost: {float(cost)} | Grads: {grads_list}')
        grads_tuple = tuple(ms.Tensor([g]) for g in grads_list)
        optimizer(grads_tuple)