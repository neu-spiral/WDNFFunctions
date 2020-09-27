import math
import numpy as np
import os
# import sympy as sp
from ProblemInstances import derive
# from scipy.misc import derivative
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# def derive(function_type, degree):
#     """Helper function to create derivatives list of Taylor objects. Given the
#     degree and the center of the Taylor expansion with the type of the functions
#     returns the value of the function's derivative at the given center point.
#     """
#     if function_type == np.log1p:
#         def derived_log(x):
#             return (((-1.0) ** (degree - 1)) * math.factorial(degree - 1)) / ((1.0 + x) ** degree)
#         if degree == 0:
#             return np.log1p  # log1p(x) is ln(x+1)
#         else:
#             return derived_log


def estimate_grad(fun, x, delta):
    """ Given a real-valued function fun, estimate its gradient numerically.
    """
    grad = (fun(x + delta) - fun(x))/delta
    return grad

# def estimateGrad(fun,x,delta):
#     """ Given a real-valued function fun, estimate its gradient numerically.
#     """
#     d = len(x)
#     grad = np.zeros(d)
#     for i in range(d):
#         e = np.zeros(d)
#         e[i] = 1.0
#         grad[i] = (fun(x+delta*e) - fun(x))/delta
#     return grad


if __name__ == "__main__":
    n = 1000000  # number of inputs
    f = np.log1p
    max_degree = 11
    y = (np.random.rand(n)).tolist()  # input list/array, uniformly distributed random values between [0, 1]
    # x = sp.Symbol('x')
    # f = sp.ln(x + 1)
    degrees = range(1, max_degree)
    errors = []
    for degree in degrees:
        total = 0.0
        # f_prime = sp.diff(f)  # derive(np.log1p, degree)
        # ff = sp.lambdify(x, f)
        # ff_prime = sp.lambdify(x, f_prime)
        for j in range(n):
            # if degree == 1:
            #     numerical = estimate_grad(f, y[j], 0.00001)
            # else:
            #     numerical = estimate_grad(lambda x: derive(f, x, degree-1), y[j], 0.00001)
            # print("input is: " + str(y[j]))
            # numerical = derivative(ff, y[j], dx=1e-10)  # estimate_grad(ff, y[j], 0.00001)
            # print("f is:" + str(numerical))
            # print("f' is:" + str(f_prime(y[j])) + '\n')
            total += (estimate_grad(lambda x: derive(f, x, degree-1), y[j], 0.00001) - derive(f, y[j], degree)) ** 2
            # total += (numerical - ff_prime(y[j])) ** 2
        # f = sp.diff(f)  # derive(np.log1p, degree - 1)
        errors.append(np.sqrt(total) / (1.0 * n))
        print("\nDifference between estimateGrad and derive of degree %d: " % degree + str(np.sqrt(total) / (1.0 * n)))
    plt.plot(degrees, errors)
    plt.title(r'$\ell_2$ norm of the difference between estimateGrad and derive')
    plt.xlabel("L")
    plt.ylabel(r'$||\frac{f^{(L-1)}(x + \delta) - f^{(L-1)}(x)}{\delta} - f^L(x)||_2 / n$')
    plt.show()
    output_dir = 'results/plots/diff_test/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + 'diff_error.png', bbox_inches="tight")
