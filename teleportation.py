import qiskit
import numpy as np
from qiskit_aer.backends import AerSimulator
import random


def teleport(statevector):
    Alice = qiskit.QuantumRegister(1, 'Alice')
    Bob = qiskit.QuantumRegister(1, 'Bob')
    State = qiskit.QuantumRegister(1, 'State')
    c = qiskit.ClassicalRegister(2)
    qc = qiskit.QuantumCircuit(State, Alice, Bob, c)
    qc.initialize(statevector, State)  # Initialize the 'State' qubit

    # Bell State between Alice and Bob
    qc.h(Alice)
    qc.cx(Alice, Bob)
    # Put a visual barrier
    qc.barrier()

    # Bell measurement
    qc.cx(State, Alice)
    qc.h(State)
    qc.measure(State, 0)
    qc.measure(Alice, 1)

    # Put a visual barrier
    qc.barrier()
    qc.z(Bob).c_if(c[0], 1)
    qc.x(Bob).c_if(c[1], 1)
    return qc

def phi_plus_circuit():
    q = qiskit.QuantumRegister(2, 'q')
    qc = qiskit.QuantumCircuit(q)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def phi_minus_circuit():
    q = qiskit.QuantumRegister(2, 'q')
    qc = qiskit.QuantumCircuit(q)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def psi_plus_circuit():
    q = qiskit.QuantumRegister(2, 'q')
    qc = qiskit.QuantumCircuit(q)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    return qc

def psi_minus_circuit():
    q = qiskit.QuantumRegister(2, 'q')
    qc = qiskit.QuantumCircuit(q)
    qc.x(0)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    return qc


def bell_measurement_circuit():
    q = qiskit.QuantumRegister(2, 'Bell State')
    c = qiskit.ClassicalRegister(2)
    qc = qiskit.QuantumCircuit(q, c)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
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


    # Draw and save the circuits needed for the report.
    phi_plus_circuit().draw(output='mpl').savefig("figures/phi_plus.png")
    phi_minus_circuit().draw(output='mpl').savefig("figures/phi_minus.png")
    psi_plus_circuit().draw(output='mpl').savefig("figures/psi_plus.png")
    psi_minus_circuit().draw(output='mpl').savefig("figures/psi_minus.png")
    teleport(statevector).draw(output='mpl').savefig("figures/teleport.png")
    bell_measurement_circuit().draw(output='mpl').savefig("figures/bell_measurement.png")






