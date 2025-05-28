# --*-- conding:utf-8 --*--
# @time:5/27/25 18:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:h2o_ground_state.py

#!/usr/bin/env python3
import os
import time
import json
import numpy as np
from scipy.optimize import minimize
from cqlib import TianYanPlatform
from cqlib.circuits import Circuit, Parameter

# Configuration
LOGIN_KEY = os.getenv('CQLIB_LOGIN_KEY', 'YOUR_LOGIN_KEY')
MACHINE_NAME = 'tianyan176'
NUM_SHOTS = 2048
MAX_ITER = 50
BATCH_SIZE = 50

# Load Hamiltonian from JSON
def load_hamiltonian(path='h2o_hamiltonian.json'):
    with open(path) as f:
        data = json.load(f)
    return list(data.items())

# Build parameterized ansatz: RY rotations + chain CZ
def build_ansatz(params, num_qubits):
    circ = Circuit(qubits=list(range(num_qubits)),
                   parameters=[Parameter(f'theta{i}') for i in range(len(params))])
    for idx, theta in enumerate(circ.parameters):
        circ.ry(idx, theta)
    for i in range(num_qubits - 1):
        circ.cz(i, i + 1)
    return circ

# Build measurement circuit for a single Pauli term
def build_measurement(term, num_qubits):
    circ = Circuit(qubits=list(range(num_qubits)))
    for idx, op in enumerate(term):
        if op == 'X':
            circ.h(idx)
        elif op == 'Y':
            circ.s(idx, dagger=True)
            circ.h(idx)
    circ.measure_all()
    return circ.qcis

# Compute expectation value from counts
def compute_expectation(counts, term):
    total = sum(counts.values()) or 1
    exp = 0.0
    for bitstr, count in counts.items():
        eigen = 1
        for idx, op in enumerate(term):
            if op == 'Z' and bitstr[::-1][idx] == '1':
                eigen *= -1
        exp += eigen * count
    return exp / total

def main():
    # Initialize platform and select machine
    platform = TianYanPlatform(login_key=LOGIN_KEY)
    devices = platform.query_quantum_computer_list()
    print('Available machines:')
    for device in devices:
        print(device)
    platform.set_machine(MACHINE_NAME)

    # Load Hamiltonian and determine qubit count
    hamiltonian = load_hamiltonian()
    num_qubits = max(int(c) for term, _ in hamiltonian for c in term if c.isdigit()) + 1

    # Define VQE energy function
    def vqe_energy(params):
        ansatz = build_ansatz(params, num_qubits)
        bound = ansatz.assign_parameters({str(p): v for p, v in zip(ansatz.parameters, params)})
        ans_qcis = bound.qcis

        circuits = []
        coeffs = []
        terms = []
        for term, coeff in hamiltonian:
            meas_qcis = build_measurement(term, num_qubits)
            circuits.append(ans_qcis + '\n' + meas_qcis)
            coeffs.append(coeff)
            terms.append(term)
            if len(circuits) >= BATCH_SIZE:
                break

        job_ids = platform.submit_experiment(circuit=circuits, num_shots=NUM_SHOTS)
        time.sleep(5)
        results = platform.query_experiment(query_id=job_ids,
                                            max_wait_time=120,
                                            sleep_time=5)

        energy = 0.0
        for res, coeff, term in zip(results, coeffs, terms):
            probs = res.get('probability', {})
            counts = {bs: int(p * NUM_SHOTS) for bs, p in probs.items()}
            energy += coeff * compute_expectation(counts, term)
        return energy

    # Run optimization
    init_params = np.random.rand(num_qubits)
    result = minimize(vqe_energy,
                      init_params,
                      method='COBYLA',
                      options={'maxiter': MAX_ITER})

    print(f"Optimized energy (Hartree): {result.fun:.6f}")
    print(f"Optimal parameters: {result.x}")

if __name__ == '__main__':
    main()
