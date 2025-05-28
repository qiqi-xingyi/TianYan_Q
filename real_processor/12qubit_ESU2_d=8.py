# --*-- conding:utf-8 --*--
# @time:5/28/25 16:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:12qubit_ESU2_d=8.py

import logging
import sys
import os
from cqlib import TianYanPlatform
from cqlib.circuits import Circuit, Parameter
import numpy as np
from cqlib.utils import LaboratoryUtils

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = f"log/{script_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)

login_key = "/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA="


def build_efficient_su2(qubits, depth):
    # generate parameters
    params = []
    for layer in range(depth):
        for q in qubits:
            params.append(Parameter(f"theta_ry_L{layer}_Q{q}"))
            params.append(Parameter(f"theta_rz_L{layer}_Q{q}"))

    circ = Circuit(qubits=qubits, parameters=params)
    it = iter(circ.parameters)

    for layer in range(depth):
        # parameterized rotations
        for q in qubits:
            ry = next(it)
            rz = next(it)
            circ.h(q)
            circ.rx(q, ry)
            circ.h(q)
            circ.rz(q, rz)

        # chain entanglers
        for i in range(len(qubits) - 1):
            circ.cx(qubits[i], qubits[i+1])

    circ.measure_all()
    return circ

if __name__ == '__main__':
    platform = TianYanPlatform(login_key=login_key)
    platform.set_machine("tianyan504")

    qubits = [121,135,149,163,177,191,205,219,233,247,261,275,289]
    logging.info(f"Qubit_num: {len(qubits)}")

    depth = 3
    effsu2 = build_efficient_su2(qubits, depth)

    # bind random parameters
    values = np.random.rand(len(effsu2.parameters))
    bound = effsu2.assign_parameters({str(p): v for p, v in zip(effsu2.parameters, values)})

    logging.info("Bound QCIS:\n%s", bound.qcis)

    query_id_single = platform.submit_experiment(
        circuit=bound.qcis,
        num_shots=5000,
    )
    logging.info(f'query_id: {query_id_single}')

    exp_result = platform.query_experiment(
        query_id=query_id_single,
        max_wait_time=120,
        sleep_time=5
    )

    for res_name, res_data in exp_result[0].items():
        logging.info(f"{res_name} : {res_data}")

    lu = LaboratoryUtils()
    probability_part = lu.readout_data_to_state_probabilities_part(result=exp_result[0])
    logging.info(f"Probability of result: {probability_part}")