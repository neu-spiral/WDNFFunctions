import math
import numpy as np
import sympy as sp
from scipy.misc import derivative


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
    n = 100  # number of inputs
    y = (np.random.rand(n)).tolist()  # input list/array
    total = 0.0
    x = sp.Symbol('x')
    f = sp.ln(x + 1)
    for degree in range(1, 100):
        f_prime = sp.diff(f)  # derive(np.log1p, degree)
        ff = sp.lambdify(x, f)
        ff_prime = sp.lambdify(x, f_prime)
        for j in range(n):
            # print("input is: " + str(y[j]))
            numerical = derivative(ff, y[j], dx=1e-10)  # estimate_grad(ff, y[j], 0.00001)
            # print("f is:" + str(numerical))
            # print("f' is:" + str(f_prime(y[j])) + '\n')
            total += (numerical - ff_prime(y[j])) ** 2
        f = sp.diff(f)  # derive(np.log1p, degree - 1)
        print("Numerical error between derivatives of degree %d: " % degree + str(np.sqrt(total) / n) + '\n')
    # print(str(sym.diff((1 + x) ** (-1))))
