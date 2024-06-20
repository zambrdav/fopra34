from dqpt_timeevolution import evolve_symmetric
import qiskit
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Literal

def time_evolution_circuit(dt, t, N, g):
    """
    This function makes the time evolution and returns the state vector
    """
    timesteps = int(t / dt)
    circuit = qiskit.QuantumCircuit(N)
    for _ in range(timesteps):
        # evolve_basic(circuit, g, dt)
        evolve_symmetric(circuit, g, dt)
    return circuit



def measure_overlap(
        N: int, circuit: qiskit.QuantumCircuit,
        state: Literal["UP", "DOWN", "up", "down"], shots=1000
) -> float:

    backend = AerSimulator()
    overlap = 0
    circuit.measure_all()
    result = backend.run(circuit, shots=shots, memory=True).result()
    memory = result.get_memory(circuit)
    for m in memory:
        if state == "down" or state == "DOWN":
            if m == '1'*N:
                overlap += 1
        elif state == "up" or state == "UP":
            if m == '0'*N:
                overlap += 1
        else:
            raise ValueError("Invalid ground state")
    overlap = overlap / shots

    return overlap






def measure_magnetization(shots: int, circuit: qiskit.QuantumCircuit):
    """
    This function measures the magnetization of a state vector
    """
    magnetization_results = []
    # backend for simulation
    backend = AerSimulator()
    # run the circuit
    circuit.measure_all()
    result = backend.run(circuit, shots=shots, memory=True).result()
    memory = result.get_memory(circuit)
    for m in memory:
        magnetization = 0
        for i in m:
            if i == '0':
                magnetization += 1
            else:
                magnetization -= 1
        magnetization = magnetization / len(m)
        magnetization_results.append(magnetization)
    magnetization_average = sum(magnetization_results) / len(magnetization_results)
    return magnetization_average





if __name__ == "__main__":


    # define the parameters
    N = 10
    g = 2
    t_max = 4
    dt = 0.01
    shots = 5000
    timesteps = np.linspace(0, t_max, 1000)
    overlap_0 = [measure_overlap(N=N, circuit=time_evolution_circuit(dt, t, N, g), state="down", shots=shots) for t in timesteps]
    overlap_1 = [measure_overlap(N=N, circuit=time_evolution_circuit(dt, t, N, g), state="up", shots=shots) for t in timesteps]

    loschmidt_rate = [-np.log(overlap_0[i] + overlap_1[i]) / N for i in range(len(timesteps))]
    magnetization = [measure_magnetization(shots=shots, circuit=time_evolution_circuit(dt, t, N, g)) for t in timesteps]

    # plot the results
    plt.figure()
    plt.plot(timesteps, loschmidt_rate, label="Loschmidt rate") 
    plt.plot(timesteps, magnetization, label="Magnetization")
    plt.xlabel("Time")
    plt.legend()
    plt.title(f"Loschmidt rate and magnetization for N={N} g={g}")
    plt.savefig("figures/loschmidt_rate_and_magnetization_measurements.png")



    """
    for shots in tqdm(shots_options):
        overlap_0 = [measure_overlap(N=N, circuit=time_evolution_circuit(dt, t, N, g), state="down", shots=shots) for t in timesteps]
        overlap_1 = [measure_overlap(N=N, circuit=time_evolution_circuit(dt, t, N, g), state="up", shots=shots) for t in timesteps]
        loschmidt = [-np.log(overlap_0[i] + overlap_1[i]) / N for i in range(len(timesteps))]
        loschmidt_rates[shots_options.index(shots)] = loschmidt

    # plot the results
    plt.figure()
    for i, loschmidt in enumerate(loschmidt_rates):
        plt.plot(timesteps, loschmidt, label=f"shots = {shots_options[i]}")
    plt.xlabel("Time")
    plt.ylabel("Loschmidt rate")
    plt.legend()
    plt.title(f"Loschmidt rate for N={N} g={g}")
    plt.savefig("figures/loschmidt_rate_measurements.png")
    

    for shots in tqdm(shots_options):
        magn = [measure_magnetization(shots=shots, circuit=time_evolution_circuit(dt, t, N, g)) for t in timesteps]
        magnetizations[shots_options.index(shots)] = magn

    # plot the results
    plt.figure()
    for i, magn in enumerate(magnetizations):
        plt.plot(timesteps, magn, label=f"shots = {shots_options[i]}")
    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.title(f"Magnetization for N={N} g={g}")
    plt.savefig("figures/magnetization_measurements.png")
    """





