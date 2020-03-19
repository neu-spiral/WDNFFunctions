from helpers import save, load
from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation, derive
from time import time
import argparse
import logging
import numpy as np
import os
import pickle
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random rewards dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problemType', default='IM', type=str, help='Type of the problem instance',
                        choices=['DR', 'QS', 'FL', 'IM'])
    parser.add_argument('--input', default='datasets/one_graph_file', type=str,
                        help='Data input for the InfluenceMaximization problem')
    parser.add_argument('--testMode', default='noTest', type=str, help='Tests the quality of the estimations from '
                        'different aspects', choices=['noTest', 'differentiation', 'estimation'])
    # parser.add_argument('--fractionalVector', type=dict,
    #                     help='If testMode is selected, checks the quality of the estimations according to this '
    #                          'fractional vector')
    # parser.add_argument('--cascades', default=1000, type=int,
    #                     help='Number of cascades used in the Independent Cascade model')
    # parser.add_argument('--p', default=0.02, type=float, help='Infection probability')
    # parser.add_argument('--rewardsInput', default="rewards.txt", help='Input file that stores rewards')
    # parser.add_argument('--partitionsInput', default="givenPartitions.txt", help='Input file that stores partitions')
    # parser.add_argument('--typesInput', default="types.txt",
    #                     help='Input file that stores targeted partitions of the ground set')
    parser.add_argument('--constraints', default=4, type=int,
                        help='Constraints dictionary with {type:cardinality} pairs')
    parser.add_argument('--estimator', default='sampler', type=str, help='Type of the estimator',
                        choices=['polynomial', 'sampler', 'samplerWithDependencies'])
    parser.add_argument('--iterations', default=10, type=int,
                        help='Number of iterations used in the Frank-Wolfe algorithm')
    parser.add_argument('--degree', default=10, type=int, help='Degree of the polynomial estimator')
    parser.add_argument('--center', default=0.5, type=float,
                        help='The point around which Taylor approximation is calculated')
    parser.add_argument('--samples', default=100, type=int,
                        help='Number of samples used to calculate the sampler estimator')
#    parser.add_argument('--timeOutput', default = "sampler_time.txt",
    #    help = 'File in which time of each iteration is stored')
#    parser.add_argument('--objectiveOutput', default = "sampler_obj.txt",
    #    help = 'File in which objective at each iteration is stored')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    directory_output = "results/continuous_greedy/"
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
    logging.info('...output directory is created...')

#    rewards = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.6, 5: 0.5, 6: 0.4} #{x_i: r_i} pairs
#    givenPartitions = {'fruits': (1, 5), 'things': (2, 3), 'actions': (4, 6)} #{P_i: (x_j)} pairs where x_j in P_i
#    types = {1: 'noun', 2: 'noun', 3: 'noun', 4: 'verb', 5: 'noun', 6: 'verb'} #{x_i: type} pairs
#    k_list = {'verb': 1, 'noun': 2}

    if args.problemType == 'DR':
        rewards = eval(open(args.rewardsInput, 'r').read())
        givenPartitions = eval(open(args.partitionsInput, 'r').read())
        types = eval(open(args.typesInput, 'r').read())
        k_list = args.constraints
        newProblem = DiversityReward(rewards, givenPartitions, log, types, k_list)

    if args.problemType == 'QS':
        pass

    if args.problemType == 'FL':
        pass

    if args.problemType == 'IM':
        logging.info('Reading edge lists...')
        graphs = load(args.input)
        logging.info('...just read %d edge list' % (len(graphs)))
#        numOfNodes = G.number_of_nodes()
#        numOfEdges = G.number_of_edges()
#        logging.info('...done. Created a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

#        logging.info('Creating cascades...')
#        newG = DiGraph()
#        newG.add_nodes_from(G.nodes())
#        graphs = [newG] * args.cascades
#        for cascade in range(args.cascades):
#            choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < args.p, ] * 2).transpose()
#            chosen_edges = np.extract(choose, G.edges())
#            chosen_edges = zip(chosen_edges[0::2], chosen_edges[1::2])
#            graphs[cascade].add_edges_from(chosen_edges)
#        logging.info('...done. Created %d cascades with %s infection probability.' % (len(graphs), args.p))

        logging.info('Defining an InfluenceMaximization problem...')
        newProblem = InfluenceMaximization(graphs, args.constraints)
        logging.info('...done. %d seeds will be selected' % args.constraints)
        output = directory_output + args.problemType + "test_case_diff_samples" + args.estimator + "_" + str(args.iterations) \
                                  + "_FW"
    if args.testMode == 'noTest':
        if args.estimator == 'polynomial':
            logging.info('Initiating the Continuous Greedy algorithm using Polynomial Estimator...')
            y, track, bases = newProblem.polynomial_continuous_greedy(args.center, args.degree, int(args.iterations))
            sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
            output += "_degree_" + str(args.degree) + "_around_" + str(args.center)

        if args.estimator == 'sampler':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator...')
            y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations)
            # output += "_" + str(args.samples) + "samples"
            sys.stderr.write("number of samples: " + str(args.samples) + '\n')
            sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
            sys.stderr.write("y is: " + str(y) + '\n')

        if args.estimator == 'samplerWithDependencies':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator with Dependencies...')
            y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations, newProblem.dependencies)
            sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
            output += "_" + str(args.samples) + "samples"

        if os.path.exists(output):
            results = load(output)
            # results.append((args.constraints, track[args.iterations - 1][0], newProblem.utility_function(y)))
            results.append((args.samples, y, newProblem.utility_function(y)))
        else:
            # results = [(args.constraints, track[args.iterations - 1][0], newProblem.utility_function(y))]
            results = [(args.samples, y, newProblem.utility_function(y))]
        save(output, results)

    elif args.testMode == 'estimation':
        y = {1: 0.5, 2: 0.5, 3: 0.5}
        # y = {1: 0.0, 2: 0.0, 3: 0.0}
        # y = {1: 0.02881619988067241, 2: 0.8720933356599558, 3: 0.9149300322150012}
        out = 0.0
        for x1 in range(2):
            for x2 in range(2):
                for x3 in range(2):
                    x = {1: x1, 2: x2, 3: x3}
                    if x1 == 0 and x2 == 0 and x3 == 0:
                        out += newProblem.utility_function(x) * (1.0 - y[1]) * (1.0 - y[2]) * (1.0 - y[3])
                    elif x1 == 0 and x2 == 0 and x3 == 1:
                        out += newProblem.utility_function(x) * (1.0 - y[1]) * (1.0 - y[2]) * y[3]
                    elif x1 == 0 and x2 == 1 and x3 == 0:
                        out += newProblem.utility_function(x) * (1.0 - y[1]) * y[2] * (1.0 - y[3])
                    elif x1 == 0 and x2 == 1 and x3 == 1:
                        out += newProblem.utility_function(x) * (1.0 - y[1]) * y[2] * y[3]
                    elif x1 == 1 and x2 == 0 and x3 == 0:
                        out += newProblem.utility_function(x) * y[1] * (1.0 - y[2]) * (1.0 - y[3])
                    elif x1 == 1 and x2 == 0 and x3 == 1:
                        out += newProblem.utility_function(x) * y[1] * (1.0 - y[2]) * y[3]
                    elif x1 == 1 and x2 == 1 and x3 == 0:
                        out += newProblem.utility_function(x) * y[1] * y[2] * (1.0 - y[3])
                    else:
                        out += newProblem.utility_function(x) * y[1] * y[2] * y[3]
        sys.stderr.write("multilinear relaxation is: " + str(out) + '\n')
        if args.estimator == 'polynomial':
            poly_output = directory_output + args.problemType + '_1_graph_y0.5' + '_poly0.5'
            start = time()
            poly_grad, poly_estimation = newProblem.get_polynomial_estimator(args.center, args.degree)\
                .estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated grad is: " + str(poly_grad) + '\n')
            sys.stderr.write("estimated value of the function is: " + str(poly_estimation) + '\n')
            if os.path.exists(poly_output):
                poly_results = load(poly_output)
                poly_results.append((elapsed_time, args.degree, poly_estimation, out))
            else:
                poly_results = [(elapsed_time, args.degree, poly_estimation, out)]
            save(poly_output, poly_results)

        if args.estimator == 'sampler':
            sampler_output = directory_output + args.problemType + '_1_graph_y0.5' + '_samp'
            start = time()
            sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples)\
                                                         .estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
            if os.path.exists(sampler_output):
                sampler_results = load(sampler_output)
                sampler_results.append((elapsed_time, args.samples, sampler_estimation, out))
            else:
                sampler_results = [(elapsed_time, args.samples, sampler_estimation, out)]
            save(sampler_output, sampler_results)

        if args.estimator == 'samplerWithDependencies':
            sampler_output = directory_output + args.problemType + '_1_graph_y0.5' + '_samp_with_dep_estimation'
            start = time()
            sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples, newProblem.dependencies)\
                                                         .estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
            if os.path.exists(sampler_output):
                sampler_results = load(sampler_output)
                sampler_results.append((elapsed_time, args.samples, sampler_estimation, out))
            else:
                sampler_results = [(elapsed_time, args.samples, sampler_estimation, out)]
            save(sampler_output, sampler_results)

    elif args.testMode == 'differentiation':
        def estimate_grad(fun, x, delta):
            """ Given a real-valued function fun, estimate its gradient numerically.
            """
            d = len(x)
            grad = np.zeros(d)
            for i in range(d):
                e = np.zeros(d)
                e[i] = 1.0
                grad[i] = (fun(x + delta * e) - fun(x)) / delta
            return grad
        n = 1000
        y = np.random.rand(n).tolist()
        total = 0.0
        f_prime = np.log1p
        for degree in range(100):
            for j in range(n):
                numerical = estimate_grad(f_prime, y[j], 0.001)
                f_prime = derive(np.log1p, y[j], degree)
                total += abs(numerical - f_prime)
            sys.stdout.write("Numerical error between derivatives of degree %d: " % degree +
                             str(total/(n * 1.0)) + '\n')





#    if args.problemType == 'IM':
#        objective = 0.0
#        time_list = []
#        obj_list = []
#        for item in track.keys():
#            for graph in range(newProblem.instancesSize):
#                for node in newProblem.groundSet:
#                    objective += 1 - newProblem.wdnf_dict[graph][node](track[item][1])
#                objective += (1.0 / newProblem.instancesSize) * np.log1p((1.0 / newProblem.graphSize) * objective)
#            time_list.append(track[item][0])
#            obj_list.append(objective)


#    timeOutput = output + "_time"
#    f = open(timeOutput, "w")
#    f.write(str(time_list))
#    f.close()

#    objectiveOutput = output + "_utilities"
#    f = open(objectiveOutput, "w")
#    f.write(str(obj_list))
#    f.close()
#        print('(Sampler) Time elapsed: ' + str(j[0]) + '    Objective is: ' + str(objective) +
#        '   Gradient is:  ' + str(j[2]))
#
#    plt.plot(time_list1, obj_list1, 'r^', time_list2, obj_list2, 'g^')
#    plt.show()


# Test segment for testing the gradients for identity function starts
    # newSamplerEstimator = newProblem.getSamplerEstimator(100)
    # newPolynomialEstimator = newProblem.getPolynomialEstimator(0, 4)

#
#     def realGrad(wdnf_list, y, epsilon = 0.00001):
#         grad = dict.fromkeys(y.iterkeys(), 0.0)
#         for i in y.keys():
#             x1 = y.copy()
#             x1[i] += epsilon
#             grad1 = 0
#
#             x0 = y.copy()
#             grad0 = 0
#             for wdnf_instance in wdnf_list:
#                 grad1 += wdnf_instance(x1)
#                 grad0 += wdnf_instance(x0)
#             grad[i] = (grad1 - grad0) / epsilon
#         return grad
#
#
#     def l2norm(grad1, grad2):
#         sum = 0.0
#         for key in grad1:
#             sum += (grad1[key] - grad2[key])**2
#         return np.sqrt(sum)
#
#     for i in range(100):
#         y = dict()
#         sum = 0.0
#         for i in range(1, 7):
#             y[i] = np.random.rand()
#             sum += y[i]
#         y = {key: y[key] / sum for key in y.keys()}
#         print('random y: ' + str(y))
#         samplerGradient = newSamplerEstimator.estimate(y)
#         polynomialGradient = newPolynomialEstimator.estimate(y)
#         realGradient = realGrad(newProblem.wdnf_list, y.copy())
#
#         print('||SamplerGrad - polyGrad|| = ' + str(l2norm(samplerGradient, polynomialGradient)))
#         print('||realGradient - samplerGrad|| = ' + str(l2norm(realGradient, samplerGradient)))
#         print('||realGradient - polyGrad|| = ' + str(l2norm(realGradient, polynomialGradient)))
#
# ##Test segment for testing the gradients ends
