
from dqpt_timeevolution import time_evolution
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




def individual_loschmidt_rate(N: int, g: float, t: float, state: Literal["UP", "DOWN", "up", "down"], dt: float = 0.1) -> float:
    """
    Calculate the Loschmidt rate for a single ground state.
    """
    # Get the state vector
    state_vector = time_evolution(dt, t, N, g)
    dim = 2**N

    # Calculate the overlap
    if state == "down" or state == "DOWN":
        # If we want to calculate with respect to the all down ground state, that is the projection
        #  of the state vector onto the all down state --> first coefficient
        overlap = np.abs(state_vector[0])**2
    elif state == "up" or state == "UP":
        # If we want to calculate with respect to the all up ground state, that is the projection
        #  of the state vector onto the all up state --> last coefficient
        overlap = np.abs(state_vector[dim - 1])**2
    else:
        raise ValueError("Invalid ground state")
    # Calculate the Loschmidt rate
    loschmidt_rate = -np.log(overlap) / N
    return loschmidt_rate

def loschmidt_rate(N: int, g: float, t: float, dt: float = 0.1) -> float:
    """
    Calculate the Loschmidt rate for the all down ground state.
    """
    dim = 2**N
    # Get the state vector
    state_vector = time_evolution(dt, t, N, g)
    # Calculate the overlap
    overlap_0 = np.abs(state_vector[0])**2
    overlap_1 = np.abs(state_vector[dim-1])**2

    total_overlap = overlap_0 + overlap_1
    # Calculate the Loschmidt rate
    loschmidt_rate = -np.log(total_overlap)/ N
    return loschmidt_rate








if __name__ == "__main__":

    system_sizes = [6, 8, 10, 12]
    g = 2
    t_max = 4
    dt = 0.01
    time_steps = np.linspace(0, t_max, 1000)

    g_values = [0.5, 0.75, 1, 1.25, 1.5]


    for N in tqdm(system_sizes):

        lodschmidt_rate_0 = [individual_loschmidt_rate(N=N, g=g, t=t, state="down", dt=dt) for t in time_steps]
        lodschmidt_rate_1 = [individual_loschmidt_rate(N=N, g=g, t=t, state="up", dt=dt) for t in time_steps]
        lodschmidt_rate = [loschmidt_rate(N=N, g=g, t=t, dt=dt) for t in time_steps]

        # Plot all three lines
        plt.plot(time_steps, lodschmidt_rate_0, label=r'$\lambda_{0}$', color="blue", linestyle="--")
        plt.plot(time_steps, lodschmidt_rate_1, label=r'$\lambda_{1}$', color="red", linestyle="--")
        plt.plot(time_steps, lodschmidt_rate, label=r'$\lambda$', color="black")
        plt.legend()
        # set the maximum value of the y-axis to 2
        plt.ylim(0, 2)
        # Insert a grid
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Loschmidt rate")
        plt.title(f"Loschmidt rate for N={N}")
        plt.savefig(f"figures/loschmidt_rate_{N}.png")
        plt.clf()
        
        



    for g_val in tqdm(g_values):
        N = 8
        lodschmidt_rate_0 = [individual_loschmidt_rate(N=N, g=g_val, t=t, state="down", dt=dt) for t in time_steps]
        lodschmidt_rate_1 = [individual_loschmidt_rate(N=N, g=g_val, t=t, state="up", dt=dt) for t in time_steps]
        lodschmidt_rate = [loschmidt_rate(N=N, g=g_val, t=t, dt=dt) for t in time_steps]

        # Plot all three lines
        plt.plot(time_steps, lodschmidt_rate_0, label=r'$\lambda_{0}$', color="blue", linestyle="--")
        plt.plot(time_steps, lodschmidt_rate_1, label=r'$\lambda_{1}$', color="red", linestyle="--")
        plt.plot(time_steps, lodschmidt_rate, label=r'$\lambda$', color="black")
        plt.legend()
        # set the maximum value of the y-axis to 2
        plt.ylim(0, 2)
        # Insert a grid
        plt.grid()
        plt.xlabel("Time[s]")
        plt.ylabel("Loschmidt rate")
        plt.title(f"Loschmidt rate for g={g_val}")
        plt.savefig(f"figures/loschmidt_rate_g_{g_val}.png")
        plt.clf()




