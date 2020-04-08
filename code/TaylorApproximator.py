import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.misc import derivative


def taylor(fun, degree, center):
    estimated_fun = [((fun.diff(x, i).subs(x, center)) * (x - center)**i) / math.factorial(i)
                     for i in range(degree + 1)]
    estimated_fun = sum(estimated_fun)
    return estimated_fun


if __name__ == "__main__":
    x = sp.Symbol('x')
    f = sp.ln(x + 1)
    x_lims = [0, 1]
    x1 = np.linspace(x_lims[0], x_lims[1], 1000)
    y1 = []
    for j in range(1, 5):
        fun = taylor(f, j, 0.0)
        y1 = [fun.subs(x, k) for k in x1]
        error = [abs(f.subs(x, m) - fun.subs(x, m)) for m in x1]
        error = sum(error) / len(x1)
        print('Average error between the function and the Taylor expansion at n = ' + str(j) + ' is ' + str(error))
        plt.plot(x1, y1, label='order' + str(j))
        y1 = []
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
    plt.savefig('results/plots/TaylorApproxOfLogDegree1to5Around00.png')
