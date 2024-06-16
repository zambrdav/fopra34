import qiskit
from qiskit.circuit.library import RXGate, RZZGate
from qiskit_aer.primitives import Estimator
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from exact_diagonalization import magn_exact_diagonalization
import matplotlib.pyplot as plt
from tqdm import tqdm


def A_operator(circuit: qiskit.QuantumCircuit, dt: float, g: float):
    x_rot = RXGate(-(g * dt))
    for i in range(circuit.num_qubits):
        circuit.append(x_rot, [i])


def B_operator(circuit: qiskit.QuantumCircuit, dt: float):
    # Assert if the number of qubits is even
    assert circuit.num_qubits % 2 == 0
    zz_rot = RZZGate(-dt)
    for i in range(0, (circuit.num_qubits//2 - 2) + 1):
        circuit.append(zz_rot, [2*i + 1, 2*i + 2])
    for i in range(0, (circuit.num_qubits//2 - 1) + 1):
        circuit.append(zz_rot, [2*i, 2*i + 1])


def evolve_basic(circ: qiskit.QuantumCircuit, g: float, dt: float):
    B_operator(circ, dt)
    A_operator(circ, dt, g)


def evolve_symmetric(circ: qiskit.QuantumCircuit, g: float, dt: float):
    A_operator(circ, dt, g/2)
    B_operator(circ, dt)
    A_operator(circ, dt, g/2)


def calculate_magnatization(dt, t, N, g):
    timesteps = int(t / dt)
    circuit = qiskit.QuantumCircuit(N)
    for _ in range(timesteps):
        #evolve_basic(circuit, g, dt)
        evolve_symmetric(circuit, g, dt)
    observable = magnetization_operator(N)

    result = Estimator().run(circuit, observable).result().values[0]
    return result


def magnetization_operator(dim):
    magnetization_list = []
    string_list = ["I" for _ in range(dim)]
    for i in range(dim):
        string_list[i] = "Z"
        string = "".join(string_list)
        magnetization_list.append((string, 1/dim))
        string_list[i] = "I"

    magnetization = SparsePauliOp.from_list(magnetization_list)
    return magnetization




if __name__ == "__main__":
    N = 10
    g = 2
    t = 5
    vectors = []
    dt_values = [0.5, 0.3, 0.1, 0.01]
    COLORS = {0.5: "red", 0.3: "blue", 0.1: "green", 0.01: "violet"}
    m_z_result = {dt: [] for dt in dt_values}
    diff_result = {dt: [] for dt in dt_values}

    time_values = np.linspace(start=0, stop=t, num=100)
    exact_m_z = magn_exact_diagonalization(N, g, t, len(time_values))

    for dt in tqdm(dt_values):
        m_z = []
        for t in time_values:
            m_z.append(calculate_magnatization(dt=dt, t=t, N=N, g=g))
        m_z_result[dt] = m_z
        diff = np.abs(exact_m_z - np.array(m_z))
        diff_result[dt] = diff
        # To avoid the errors due to dividing close to 0, delete values
        print(f"dt: {dt}, max diff: {max(diff)}")


    # Plot the results for the magnetization
    plt.plot(time_values, exact_m_z, label="Exact", color="black")
    for dt in dt_values:
        plt.plot(time_values, m_z_result[dt], label=f"dt: {dt}", color=COLORS[dt])
    plt.legend()
    plt.title("Magnetization for different timesteps with second order Trotterization")
    # axis label
    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    # save figure
    plt.savefig("figures/magnetization_symmetric.png")
    plt.clf()

    # Plot the results for the difference
    for dt in dt_values:
        plt.plot(time_values, diff_result[dt], label=f"dt: {dt}", color=COLORS[dt])
    plt.title("Error with second order Trotterization")
    plt.legend()
    # axis label
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    # save figure
    plt.savefig("figures/error_symmetric.png")
    plt.clf()





