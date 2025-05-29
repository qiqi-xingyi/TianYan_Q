# --*-- conding:utf-8 --*--
# @time:5/29/25 14:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:15qubit_VQE.py

from mindquantum.core import Circuit, apply, RY
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

def compute_cost(result: dict, num_qubits: int):
    probs = json.loads(result['probability'])
    zero_key = '0' * num_qubits
    return probs.get(zero_key, 0.0)


def hardware_run(circuits: list[Circuit], shots: int = 1000):
    tasks = [circ.to_qcis(parametric=False) for circ in circuits]
    query_list = platform.submit_experiment(
        circuit=tasks,
        language=QuantumLanguage.QCIS,
        num_shots=shots
    )
    results = platform.query_experiment(
        query_id=query_list,
        max_wait_time=600,
        sleep_time=5
    )
    costs = []
    for res in results:
        costs.append(compute_cost(res, num_qubits))
    return costs

def get_cost_with_grads(circuit: Circuit):
    names_e = circuit.encoder_params_name
    names_a = circuit.ansatz_params_name
    h = np.pi / 2

    def grad_ops(values_e, values_a):
        circs = []
        pr_e = dict(zip(names_e, values_e))
        base = circuit.apply_value(pr_e)

        pr_0 = dict(zip(names_a, values_a))
        circs.append(base.apply_value(pr_0))

        circs_p = []
        circs_n = []
        for i in range(len(values_a)):
            vp = deepcopy(values_a)
            vn = deepcopy(values_a)
            vp[i] += h
            vn[i] -= h
            pr_p = dict(zip(names_a, vp))
            pr_n = dict(zip(names_a, vn))
            circs_p.append(base.apply_value(pr_p))
            circs_n.append(base.apply_value(pr_n))

        circs.extend(circs_p)
        circs.extend(circs_n)

        costs = hardware_run(circs)
        cost = costs.pop(0)
        cost_p = ms.Tensor(costs[:len(values_a)])
        cost_n = ms.Tensor(costs[len(values_a):])
        grads = (cost_p - cost_n) / 2
        return cost, grads

    return grad_ops

if __name__ == '__main__':
    circ = Circuit()
    num_qubits = 15

    # Encoder layer
    for i in range(num_qubits):
        circ += RY('alpha').on(i)
    circ.as_encoder()

    # Ansatz layer
    for i in range(num_qubits):
        circ += RY('theta').on(i)
    circ.measure_all()

    # Map to physical qubits
    physical_qubits = [259, 273, 287, 301, 315, 329, 343, 357, 371, 385, 399, 413, 427, 441, 455]
    circ = apply(circ, physical_qubits)

    grad_ops = get_cost_with_grads(circ)

    theta = Parameter(ms.Tensor([0.]), name='theta')
    optim = nn.SGD([theta], learning_rate=0.5)

    login_key = '/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA='
    machine_name = 'tianyan504'
    platform = TianYanPlatform(login_key=login_key, machine_name=machine_name)

    encoder_vals = ms.Tensor([np.pi/2] * num_qubits)

    for i in range(20):
        cost, grads = grad_ops(encoder_vals, theta)
        logger.info(f'step: {i}, theta: {theta.asnumpy()}, cost: {cost}, grads: {grads.asnumpy()}')
        optim((grads,))
