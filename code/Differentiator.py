import math
import numpy as np


def derive(function_type, degree):
    """Helper function to create derivatives list of Taylor objects. Given the
    degree and the center of the Taylor expansion with the type of the functions
    returns the value of the function's derivative at the given center point.
    """
    if function_type == np.log1p:
        def derived_log(x):
            return (((-1.0) ** (degree - 1)) * math.factorial(degree - 1)) / ((1.0 + x) ** degree)
        if degree == 0:
            return np.log1p  # log1p(x) is ln(x+1)
        else:
            return derived_log


def estimate_grad(fun, x, delta):
    """ Given a real-valued function fun, estimate its gradient numerically.
    """
    grad = (fun(x + delta) - fun(x))/delta
    return grad


if __name__ == "__main__":
    n = 100
    y = (np.random.rand(n) * 1).tolist()
    print(y)
    total = 0.0
    for degree in range(1, 100):
        f_prime = derive(np.log1p, degree)
        f = derive(np.log1p, degree - 1)
        for j in range(n):
            numerical = estimate_grad(f, y[j], 0.000001)
            total = (numerical - f_prime(y[j]))
        print("Numerical error between derivatives of degree %d: " % degree + str(total) + '\n')
