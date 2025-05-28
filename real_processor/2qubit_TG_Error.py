# --*-- conding:utf-8 --*--
# @time:5/28/25 15:02
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:2qubit_TG_Error.py

import logging
import sys
import os
from cqlib import TianYanPlatform
from cqlib.circuits import Circuit, Parameter

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = f"{script_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)

login_key = "/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA="

if __name__ == '__main__':

    platform = TianYanPlatform(login_key=login_key)

    computer_list_data = platform.query_quantum_computer_list()

    for computer_data in computer_list_data:
        print(computer_data)

    platform.set_machine("tianyan504")

    circuit = Circuit(qubits=[1, 5])

    circuit.h(1)
    circuit.x(5)
    circuit.h(5)
    circuit.cz(1, 5)
    circuit.h(5)
    circuit.measure_all()

    print(circuit.qcis)

    theta = Parameter('theta')
    circuit_para = Circuit(qubits=[0], parameters=[theta])
    circuit_para.rx(0, theta)

    c1 = circuit_para.assign_parameters({'theta': 0.12})

    query_id_single = platform.submit_job(
        circuit=circuit.qcis,
        num_shots=5000,
    )

    print(f'query_id: {query_id_single}')

    exp_result = platform.query_experiment(query_id=query_id_single, max_wait_time=120, sleep_time=5)

    for res_name, res_data in exp_result[0].items():
        print(f"{res_name} : {res_data}")