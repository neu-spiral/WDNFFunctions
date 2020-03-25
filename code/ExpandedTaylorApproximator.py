import math
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from helpers import save, load
from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation, derive
from scipy.misc import derivative
# matplotlib.use('Agg')


def taylor(fun, degree, center):
    estimated_fun = [((fun.diff(x, i).subs(x, center)) * (x - center)**i) / math.factorial(i)
                     for i in range(degree + 1)]
    estimated_fun = sum(estimated_fun)
    return estimated_fun


if __name__ == "__main__":
    graphs = load('datasets/one_graph_file')
    newProblem = InfluenceMaximization(graphs, 1)
    g = newProblem.wdnf_dict[0]
    # print("g^2(x) is: " + str((g**2).coefficients))

    x = sp.Symbol('x')
    f = sp.ln(x + 1)
    # x_lims = [0, 1]
    x1 = [{1: 0.0, 2: 0.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 1.0}, {1: 0.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 1.0, 3: 1.0},
          {1: 1.0, 2: 0.0, 3: 0.0}, {1: 1.0, 2: 0.0, 3: 1.0}, {1: 1.0, 2: 1.0, 3: 0.0}, {1: 1.0, 2: 1.0, 3: 1.0}]
    # y1 = []
    for j in range(1, 100):
        expanded_f = newProblem.get_polynomial_estimator(0.5, j).my_wdnf
        # print('expanded f at degree = ' + str(j) + ' is ' + str(expanded_f.coefficients))
        fun = taylor(f, j, 0.5)
        # print(sp.expand(fun))
        # y1 = [fun.subs(x, k) for k in x1]
        error = 0.0
        for y in x1:
            # print('for x = ' + str(y))
            # print('expanded f at n = ' + str(j) + ' is ' + str(expanded_f(y)))
            # print('f at n = ' + str(j) + ' is ' + str(fun.subs(x, g(y))))
            error += abs(expanded_f(y) - fun.subs(x, g(y)))
        error = error / len(x1)
        print('Average error at n = ' + str(j) + ' is ' + str(error))
        # plt.plot(x1, y1, label='order' + str(j))
        # y1 = []
    # plt.plot(x1, np.log1p(x1), label='log(x + 1)')
    # plt.xlim(x_lims)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.grid(True)
    # plt.title('Taylor series approximation of log(x + 1)')
    # plt.show()
    # plt.savefig('results/plots/TaylorApproxOfLogDegree10to20.png')
