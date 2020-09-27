from ProblemInstances import find_derivatives
from wdnf import Taylor
import math
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import sympy as sp
# from scipy.misc import derivative


# def taylor(fun, degree, center):
#     estimated_fun = [((fun.diff(x, i).subs(x, center)) * (x - center)**i) / math.factorial(i)
#                      for i in range(degree + 1)]
#     estimated_fun = sum(estimated_fun)
#     return estimated_fun


if __name__ == "__main__":
    # x = sp.Symbol('x')
    # f = sp.ln(x + 1)
    x_lims = [0, 1]
    x1 = np.linspace(x_lims[0], x_lims[1], 1000000)
    center = 0.5
    degrees = range(1, 11)
    avg_errs = []
    for j in degrees:
        y1 = []
        # fun = taylor(f, j, 0.0)
        # y1 = [fun.subs(x, k) for k in x1]
        # error = [abs(f.subs(x, m) - fun.subs(x, m)) for m in x1]
        derivatives = find_derivatives(np.log1p, center, j)
        my_taylor = Taylor(j, derivatives, center)
        y1 = [my_taylor(m) for m in x1]
        error = [abs(np.log1p(m) - my_taylor(m)) for m in x1]
        error = sum(error) / len(x1)
        avg_errs.append(error)
        print('\nAverage error between the function and the Taylor expansion at n = ' + str(j) + ' is ' + str(error))
        plt.plot(x1, y1, label='L = ' + str(j))
    plt.plot(x1, np.log1p(x1), label='log(x + 1)')
    mean_y = sum(np.log1p(x1)) / len(x1)
    mean_x = np.expm1(mean_y)
    print("Mean value of log(x + 1) where x in [0, 1] is: " + str(mean_x))
    plt.xlim(x_lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation of log(x + 1)')
    plt.show()
    output_dir = 'results/plots/tests/taylor/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + 'TaylorApprox.png', bbox_inches="tight")
    plt.figure()
    plt.plot(degrees, avg_errs)
    plt.xlabel('L')
    plt.ylabel(r'$\frac{1}{|x|} \sum |f(x) - \hat{f_L}(x)|$')
    plt.title(r'Average error between log(x + 1) and $L^th$ Taylor Approximation')
    plt.show()
    plt.savefig(output_dir + 'AvgErrorTaylor.png', bbox_inches="tight")


