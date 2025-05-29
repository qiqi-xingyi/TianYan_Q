# --*-- conding:utf-8 --*--
# @time:5/29/25 12:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:VQE.py

from mindquantum.core import Circuit, apply, RY
from cqlib import TianYanPlatform, QuantumLanguage
from copy import deepcopy
import json
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.common.parameter import Parameter


def compute_cost(result: dict):
    return json.loads(result['probability'])['0']


def hardware_run(circs: list[Circuit], shots: int = 1000):

    tasks = [circ.to_qcis(parametric=False) for circ in circs]

    query_list = platform.submit_experiment(circuit=tasks,
                                            language=QuantumLanguage.QCIS,
                                            num_shots=shots)

    results = platform.query_experiment(query_id=query_list,
                                        max_wait_time=600,
                                        sleep_time=5)

    costs = []
    for i in range(len(circs)):
        costs.append(compute_cost(results[i]))
    return costs


def get_cost_with_grads(circuit: Circuit):

    names_e = circuit.encoder_params_name
    names_a = circuit.ansatz_params_name
    h = np.pi / 2

    def grad_ops(values_e, values_a):

        circs_p = []
        circs_n = []
        circs = []

        pr_e = dict(zip(names_e, values_e))
        circ = circuit.apply_value(pr_e)

        pr_0 = dict(zip(names_a, values_a))
        circs.append(circ.apply_value(pr_0))

        for i in range(len(values_a)):
            values_a_p = deepcopy(values_a)
            values_a_n = deepcopy(values_a)

            values_a_p[i] += h
            pr_p = dict(zip(names_a, values_a_p))
            circs_p.append(circ.apply_value(pr_p))

            values_a_n[i] -= h
            pr_n = dict(zip(names_a, values_a_n))
            circs_n.append(circ.apply_value(pr_n))

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
    circ += RY("alpha").on(0)
    circ.as_encoder()
    circ += RY("theta").on(0)
    circ.measure_all()

    physical_qubits = [13]
    circ = apply(circ, physical_qubits)

    grad_ops = get_cost_with_grads(circ)

    theta = Parameter(ms.Tensor([0.]), name="theta")
    optim = nn.SGD([theta], learning_rate=0.5)

    login_key = "/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA="
    machine_name = "tianyan504"
    platform = TianYanPlatform(login_key=login_key, machine_name=machine_name)

    for i in range(20):
        cost, grads = grad_ops(ms.Tensor([np.pi / 2]), theta)
        print("step:", i, "theta", theta.asnumpy(), "cost:", cost, "grads:", grads.asnumpy())
        optim((grads,))
