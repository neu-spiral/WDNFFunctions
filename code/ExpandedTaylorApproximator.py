import argparse
import math
import numpy as np
import random
# import sympy as sp
from helpers import save, load
from ProblemInstances import find_derivatives
from wdnf import Taylor
# from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation, derive
# from scipy.misc import derivative
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# def taylor(fun, degree, center):
#     estimated_fun = [((fun.diff(x, i).subs(x, center)) * (x - center)**i) / math.factorial(i)
#                      for i in range(degree + 1)]
#     estimated_fun = sum(estimated_fun)
#     return estimated_fun


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Module for ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problem', type=str, help='If the problem instance is created before, provide it here to save'
                                                    ' time instead of recreating it.')
    args = parser.parse_args()
    num_of_samples = 100
    # graphs = load('datasets/one_graph_file')
    newProblem = load(args.problem)  # InfluenceMaximization(graphs, 1)

    # print("g^2(x) is: " + str((g**2).coefficients))

    # x = sp.Symbol('x')
    # f = sp.ln(x + 1)
    # x_lims = [0, 1]
    vectors = []
    for n in range(num_of_samples):
        binary_vector = map(int, list(bin(random.getrandbits(newProblem.problemSize))[2:]))
        if len(binary_vector) < newProblem.problemSize:
            binary_vector = [0] * (newProblem.problemSize - len(binary_vector)) + binary_vector
        # random_list = random.getrandbits(newProblem.problemSize)
        y = dict(zip(newProblem.groundSet, binary_vector))
        vectors.append(y)
    # x1 = [{1: 0.0, 2: 0.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 1.0}, {1: 0.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 1.0, 3: 1.0},
    #       {1: 1.0, 2: 0.0, 3: 0.0}, {1: 1.0, 2: 0.0, 3: 1.0}, {1: 1.0, 2: 1.0, 3: 0.0}, {1: 1.0, 2: 1.0, 3: 1.0}]
    # y1 = []
    center = 0.5
    degrees = range(1, 11)
    avg_errs = []
    for j in degrees:
        expanded_f = newProblem.get_polynomial_estimator(0.5, j).my_wdnf
        # print('expanded f at degree = ' + str(j) + ' is ' + str(expanded_f.coefficients))
        # for g in newProblem.wdnf_dict:
        # fun = taylor(f, j, 0.5)
        # print(sp.expand(fun))
        # y1 = [fun.subs(x, k) for k in x1]
        derivatives = find_derivatives(np.log1p, center, j)
        my_taylor = Taylor(j, derivatives, center)
        error = 0.0
        for y in vectors:
            # print('for x = ' + str(y))
            # print('expanded f at n = ' + str(j) + ' is ' + str(expanded_f(y)))
            # print('f at n = ' + str(j) + ' is ' + str(fun.subs(x, g(y))))
            final_value = [(1.0 / newProblem.instancesSize) * my_taylor(newProblem.wdnf_dict[g](y)) for g in
                           range(newProblem.instancesSize)]
            final_value = sum(final_value)
            error += abs(expanded_f(y) - final_value)
        error = error / num_of_samples
        avg_errs.append(error)
        print('Average error at L = ' + str(j) + ' is ' + str(error))
    output_dir = 'results/plots/tests/taylor/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure()
    plt.plot(degrees, avg_errs)
    plt.xlabel('L')
    plt.ylabel(r'$\frac{1}{n} \sum |\tilde{f}(x) - \hat{f_L}(g(x))|$')
    plt.title(r'Average error between \tilde{f}(x) and $\hat{f_L}(g(x))$')
    plt.show()
    plt.savefig(output_dir + 'AvgErrExpTaylorApprox.png', bbox_inches="tight")