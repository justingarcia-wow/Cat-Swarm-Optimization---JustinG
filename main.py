import numpy as np
import matplotlib.pyplot as plt
import time

from cso import CatSwarmOptimization
from functions import sphere
from baseline import random_search


def main():

    dim = 2
    bounds = (-10, 10)

    num_cats = 80
    max_iter = 600

    runs = 5

    histories = []
    cso_results = []

    print("\n==============================")
    print("RUNNING CAT SWARM OPTIMIZATION")
    print("==============================\n")

    for run in range(runs):

        cso = CatSwarmOptimization(
            objective_function=sphere,
            dim=dim,
            bounds=bounds,
            num_cats=num_cats,
            max_iter=max_iter
        )

        start = time.time()

        best_pos, best_fit, history = cso.optimize()

        end = time.time()

        histories.append(history)
        cso_results.append(best_fit)

        rand_pos, rand_fit = random_search(sphere, dim, bounds)

        print(f"Run {run+1}")
        print("------------------------------")
        print(f"CSO fitness:     {best_fit:.10f}")
        print(f"Random fitness:  {rand_fit:.10f}")

        if best_fit < rand_fit:
            print("Winner: CSO")
        else:
            print("Winner: Random")

        print(f"Execution time: {end-start:.4f}s\n")

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")

    print(f"Average CSO fitness: {np.mean(cso_results):.6f}")

    # -------- PLOT --------

    histories = np.array(histories)

    for i in range(runs):
        plt.plot(histories[i], alpha=0.6, label=f"Run {i+1}")

    avg_curve = np.mean(histories, axis=0)

    plt.plot(avg_curve, linewidth=3, label="Average")

    plt.title("CSO Convergence on Sphere Function")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()