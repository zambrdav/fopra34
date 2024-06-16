import qiskit
from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RZXGate, RZGate
from itertools import product
import matplotlib.pyplot as plt
# import a noise model
from qiskit_aer.noise import NoiseModel, amplitude_damping_error


def param_circuit(a: float, b: float) -> qiskit.QuantumCircuit:
    # apply the operators
    circuit = qiskit.QuantumCircuit(2)
    circuit.append(RZXGate(theta=-2*b), [0, 1])
    circuit.append(RZXGate(theta=-2*b), [1, 0])
    circuit.append(RZGate(phi=-2*a), [0])
    circuit.append(RZGate(phi=-2*a), [1])
    return circuit



def grid_search(hamiltonian, values, noise_model=None):
    results = np.zeros((20, 20))
    for a_ind, b_ind in product(range(len(values)), repeat=2):
        a = values[a_ind]
        b = values[b_ind]
        circuit = param_circuit(a, b)
        if noise_model is None:
            estimator = Estimator()
        else:
            estimator = Estimator(
                backend_options={"noise_model": noise_model}
            )
        result = estimator.run(circuit, hamiltonian).result().values[0]
        results[a_ind, b_ind] = result
    return results

def calculate_energy(a, b):
    circuit = param_circuit(a, b)
    estimator = Estimator()
    result = estimator.run(circuits=circuit, observables=hamiltonian).result().values
    return result



if __name__ == "__main__":

    # find the optimal values for a and b:
    values = np.linspace(start=0., stop=np.pi, num=20)

    X = SparsePauliOp("X")
    Z = SparsePauliOp("Z")
    hamiltonian = -(Z ^ X) - (X ^ Z)
    results = grid_search(hamiltonian, values)
    # make a heatmap of the results and save the figure
    plt.imshow(results, origin='lower', extent=(0, np.pi, 0, np.pi))
    plt.title("Energy value for the parametrized circuit")
    plt.colorbar()
    plt.xlabel("a")
    plt.ylabel("b")
    plt.savefig("heatmap.png")
    plt.clf()

    # get the best values for a and b:
    a_opt, b_opt = np.unravel_index(np.argmin(results), results.shape)

    # run the parametrized circuit with the optimal values found:
    backend = AerSimulator(method='statevector')
    circuit = param_circuit(values[a_opt], values[b_opt])
    circuit.save_statevector()
    state = backend.run(circuit).result().get_statevector()
    # we obtian up to a phase (and some error) the state: 1/2( |01> + |10> - |11> + |00> )
    print("Optimal values: ", values[a_opt], values[b_opt])
    print("Groundstate: ", state)

    noise_model = NoiseModel()
    # excited state population is p and param_amp is gamma
    error_single = amplitude_damping_error(excited_state_population=0.2, param_amp=0.1)
    error_double = amplitude_damping_error(excited_state_population=0.2, param_amp=0.1).tensor(amplitude_damping_error(excited_state_population=0.2, param_amp=0.01))

    noise_model.add_all_qubit_quantum_error(error=error_single, instructions=["rz"])
    noise_model.add_all_qubit_quantum_error(error=error_double, instructions=["rzz"])
    results_noise = grid_search(hamiltonian, values, noise_model=noise_model)
    plt.imshow(results_noise, origin='lower', extent=(0, np.pi, 0, np.pi))
    plt.title("Energy value for the parametrized circuit with noise model")
    plt.colorbar()
    plt.xlabel("a")
    plt.ylabel("b")
    plt.savefig("heatmap_noise.png")
    plt.clf()
    # get the best values for a and b:
    a_opt_noise, b_opt_noise = np.unravel_index(np.argmin(results_noise), results_noise.shape)

    # run the parametrized circuit with the optimal values found:
    backend_noise = AerSimulator(method='statevector', noise_model=noise_model)
    circuit_noise = param_circuit(values[a_opt_noise], values[b_opt_noise])
    circuit_noise.save_statevector()
    state_noise = backend_noise.run(circuit_noise).result().get_statevector()
    print("Optimal values with noise: ", values[a_opt_noise], values[b_opt_noise])
    print("Groundstate with noise: ", state_noise)































