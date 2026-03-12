import numpy as np

def sphere(x):
    """
    Sphere function
    f(x) = sum(x_i^2)
    Global minimum at x = 0
    """
    return np.sum(x**2)