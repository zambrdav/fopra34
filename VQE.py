import qiskit
from qiskit_aer.primitives import Estimator

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from itertools import product
import matplotlib.pyplot as plt





def param_circuit(a, b):

    #operator_a = SparsePauliOp.from_list([("II", np.cos(a)), ("ZI", np.sin(a) * 1j), ("IZ", np.sin(a) * 1j)])
    #operator_b = SparsePauliOp.from_list([("II", np.cos(b)), ("ZX", np.sin(b) * 1j), ("XZ", np.sin(b) * 1j)])
    X = SparsePauliOp("X")
    Z = SparsePauliOp("Z")
    I = SparsePauliOp("I")
    op_a = (Z ^ I) + (I ^ Z)
    op_b = (Z ^ X) + (X ^ Z)
    operator_a = PauliEvolutionGate(op_a, time=-a)
    operator_b = PauliEvolutionGate(op_b, time=-b)
    # apply the operators
    circuit = qiskit.QuantumCircuit(2)
    circuit.append(operator_a, range(2))
    circuit.append(operator_b, range(2))
    return circuit



def grid_search(hamiltonian, values):
    results = np.zeros((20, 20))
    for a_ind, b_ind in product(range(len(values)), repeat=2):
        a = values[a_ind]
        b = values[b_ind]
        circuit = param_circuit(a, b)
        estimator = Estimator()
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
    #results2 = [[calculate_energy(a, b) for a in values] for b in values]
    #results2 = np.array(results2)

    # run the parametrized circuit with the optimal values found:
    plt.imshow(results)
    plt.show()





















