import qiskit
from qiskit.circuit.library import RXGate, RZZGate
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from exact_diagonalization import magn_exact_diagonalization



def evolve_basic(circ: qiskit.QuantumCircuit, g: float, dt: float):
    x_rot = RXGate(-(g * dt))
    zz_rot = RZZGate(-dt)
    for i in range(circ.num_qubits - 1):
        circ.append(x_rot, [i])
    for i in range(circ.num_qubits - 1):
        circ.append(zz_rot, [i, i + 1])


def evolve_symmetric(circ: qiskit.QuantumCircuit, g: float, dt: float):
    x_rot = RXGate(-(g * dt)/2)
    zz_rot = RZZGate(-dt)
    for i in range(circ.num_qubits - 1):
        circ.append(x_rot, [i])
    for i in range(circ.num_qubits - 1):
        circ.append(zz_rot, [i, i + 1])
    for i in range(circ.num_qubits - 1):
        circ.append(x_rot, [i])

def time_evolution(dt, t, N):
    timesteps = int(t / dt)
    circuit = qiskit.QuantumCircuit(N)
    backend = AerSimulator(method='statevector')
    for _ in range(timesteps):
        #evolve_basic(circuit, g, dt)
        evolve_symmetric(circuit, g, dt)
    circuit.save_statevector()
    vector = backend.run(circuit).result().get_statevector()
    return vector

Z = np.array([[1, 0], [0, -1]])

def magnetization(vector):
    vector = np.array(vector)
    dim = len(vector)
    N = int(np.log2(dim))
    magnetization_list = []
    string_list = ["I" for _ in range(N)]
    for i in range(N):
        string_list[i] = "Z"
        string = "".join(string_list)
        magnetization_list.append((string, 1/N))
        string_list[i] = "I"

    magnetization = SparsePauliOp.from_list(magnetization_list)
    magnetization_matrix = magnetization.to_matrix()
    return np.dot(vector.conj().T, magnetization_matrix @ vector)








if __name__ == "__main__":
    N = 10
    g = 2
    dt = 0.01
    vectors = []
    times = [0.5, 1, 1.5, 2]

    m_z = []
    exact_m_z = magn_exact_diagonalization(N, g, 2, 4)



    for t in times:
        vector = time_evolution(dt, t, N)
        vectors.append(vector)
        m_z.append(magnetization(vector).real)

    print("Magnetization: ", m_z)
    print("Exact magnetization: ", exact_m_z)




















