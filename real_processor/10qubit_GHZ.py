# --*-- conding:utf-8 --*--
# @time:5/28/25 16:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:10qubit_GHZ.py

from cqlib import TianYanPlatform
from cqlib.circuits import Circuit
import logging, sys, os
from cqlib.utils import LaboratoryUtils

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = f"log/{script_name}.log"

login_key = "/4tgPFVBopfQbgLOXE4JGnGLXOeBiTJzfDw9rrXF/wA="

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)

def build_ghz(qubits):
    circ = Circuit(qubits=qubits)
    circ.h(qubits[0])
    for i in range(len(qubits)-1):
        circ.cx(qubits[i], qubits[i+1])
    circ.measure_all()
    return circ

if __name__ == '__main__':

    platform = TianYanPlatform(login_key=login_key)
    platform.set_machine("tianyan504")

    qubits = [121, 135, 149, 163, 177, 191, 205, 219, 233, 247, 261]
    logging.info(f"Qubit_num: {len(qubits)}")

    ghz = build_ghz(qubits)


    logging.info("GHZ QCIS:\n%s", ghz.qcis)


    query_id = platform.submit_experiment(
        circuit=ghz.qcis,
        num_shots=10000,
    )
    logging.info("query_id: %s", query_id)

    exp_result = platform.query_experiment(
        query_id=query_id,
        max_wait_time=120,
        sleep_time=5
    )

    lu = LaboratoryUtils()
    probability_part = lu.readout_data_to_state_probabilities_part(result=exp_result[0])
    logging.info(f"Probability of result: {probability_part}")
