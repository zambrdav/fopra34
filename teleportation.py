import qiskit
import numpy as np
from qiskit_aer.backends import AerSimulator
import random

def teleport(statevector):
    q = qiskit.QuantumRegister(3)
    c = qiskit.ClassicalRegister(2)
    qc = qiskit.QuantumCircuit(q, c)
    qc.initialize(statevector, 0)
    # Bell State between alice and bob
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    qc.z(2).c_if(c[0], 1)
    qc.x(2).c_if(c[1], 1)
    return qc


if __name__ == "__main__":

    # create random state vector
    statevector = np.random.rand(2) + np.random.rand(2) * 1j
    statevector = statevector / np.linalg.norm(statevector)

    # create quantum circuit
    qc = teleport(statevector)
    qc.save_statevector()

    # run the quantum circuit
    sim = AerSimulator(method='statevector')
    state = sim.run(qc).result().get_statevector()
    print("Statevector after teleportation: ", state)
    print("Original statevector: ", statevector)




