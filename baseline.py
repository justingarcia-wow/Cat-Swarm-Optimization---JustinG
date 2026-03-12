import numpy as np

def random_search(objective_function, dim, bounds, iterations=1000):

    best_position = None
    best_fitness = float("inf")

    for _ in range(iterations):

        candidate = np.random.uniform(bounds[0], bounds[1], dim)

        fitness = objective_function(candidate)

        if fitness < best_fitness:
            best_fitness = fitness
            best_position = candidate

    return best_position, best_fitness