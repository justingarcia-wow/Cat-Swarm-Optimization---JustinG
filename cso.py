import numpy as np
from dataclasses import dataclass


@dataclass
class Cat:
    position: np.ndarray
    velocity: np.ndarray
    fitness: float


class CatSwarmOptimization:

    def __init__(
        self,
        objective_function,
        dim,
        bounds,
        num_cats=50,
        max_iter=400,
        mixture_ratio=0.8
    ):

        self.f = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_cats = num_cats
        self.max_iter = max_iter
        self.mixture_ratio = mixture_ratio

        self.w = 0.7
        self.c1 = 1.7
        self.cats = []
        self.best_cat = None

        for _ in range(num_cats):

            pos = np.random.uniform(bounds[0], bounds[1], dim)
            vel = np.random.uniform(-1, 1, dim)
            fit = self.f(pos)

            cat = Cat(pos, vel, fit)
            self.cats.append(cat)

            if self.best_cat is None or fit < self.best_cat.fitness:
                self.best_cat = Cat(pos.copy(), vel.copy(), fit)

    def optimize(self):

        history = []

        for iteration in range(self.max_iter):

            for cat in self.cats:

                if np.random.rand() < self.mixture_ratio:
                    self.tracing_mode(cat)
                else:
                    self.seeking_mode(cat)

                cat.fitness = self.f(cat.position)

                if cat.fitness < self.best_cat.fitness:
                    self.best_cat = Cat(
                        cat.position.copy(),
                        cat.velocity.copy(),
                        cat.fitness
                    )

            history.append(self.best_cat.fitness)

        return self.best_cat.position, self.best_cat.fitness, history

    def tracing_mode(self, cat):

        r1 = np.random.rand(self.dim)

        cat.velocity = (
            self.w * cat.velocity +
            self.c1 * r1 * (self.best_cat.position - cat.position)
        )

        cat.position = cat.position + cat.velocity

        cat.position = np.clip(
            cat.position,
            self.bounds[0],
            self.bounds[1]
        )

    def seeking_mode(self, cat):

        candidate = cat.position + np.random.normal(0, 0.2, self.dim)

        candidate = np.clip(candidate, self.bounds[0], self.bounds[1])

        if self.f(candidate) < cat.fitness:
            cat.position = candidate