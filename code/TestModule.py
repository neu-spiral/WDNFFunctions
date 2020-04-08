from ContinuousGreedy import multilinear_relaxation
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
    parser = argparse.ArgumentParser(description='Test Module for ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problemType', default='DR', type=str, help='Type of the problem instance',
                        choices=['DR', 'QS', 'FL', 'IM'])
    parser.add_argument('--input', default='datasets/epinions_20', type=str,
                        help='Data input for the InfluenceMaximization problem')
    parser.add_argument('--testMode', default=False, type=bool, help='Tests the quality of the estimations from '
                        'different aspects')
    # parser.add_argument('--fractionalVector', type=dict,
    #                     help='If testMode is selected, checks the quality of the estimations according to this '
    #                          'fractional vector')
    # parser.add_argument('--cascades', default=1000, type=int,
    #                     help='Number of cascades used in the Independent Cascade model')
    # parser.add_argument('--p', default=0.02, type=float, help='Infection probability')
    parser.add_argument('--rewardsInput', default="datasets/DR_rewards0", help='Input file that stores rewards')
    parser.add_argument('--partitionsInput', default="datasets/DR_givenPartitions0", help='Input file that stores partitions')
    parser.add_argument('--typesInput', default="datasets/DR_types0",
                        help='Input file that stores targeted partitions of the ground set')
    parser.add_argument('--constraints', default="datasets/DR_k_list0",
                        help='Constraints dictionary with {type:cardinality} pairs')
    parser.add_argument('--estimator', default='sampler', type=str, help='Type of the estimator',
                        choices=['polynomial', 'sampler', 'samplerWithDependencies'])
    parser.add_argument('--iterations', default=50, type=int,
                        help='Number of iterations used in the Frank-Wolfe algorithm')
    parser.add_argument('--degree', default=4, type=int, help='Degree of the polynomial estimator')
    parser.add_argument('--center', default=0.5, type=float,
                        help='The point around which Taylor approximation is calculated')
    parser.add_argument('--samples', default=500, type=int,
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

    if args.problemType == 'DR':
        rewards = load(args.rewardsInput)
        givenPartitions = load(args.partitionsInput)
        types = load(args.typesInput)
        k_list = load(args.constraints)
        newProblem = DiversityReward(rewards, givenPartitions, types, k_list)

    if args.problemType == 'QS':
        pass

    if args.problemType == 'FL':
        pass

    if args.problemType == 'IM':
        logging.info('Reading edge lists...')
        graphs = load(args.input)
        logging.info('...just read %d edge list' % (len(graphs)))
        logging.info('Defining an InfluenceMaximization problem...')
        newProblem = InfluenceMaximization(graphs, args.constraints)
        logging.info('...done. %d seeds will be selected' % args.constraints)

    output = directory_output + args.problemType + "_" + args.input.split("/")[-1] + "_" + args.estimator + "_" \
                              + str(args.iterations) + "_FW"

    if args.testMode is False:
        if args.estimator == 'polynomial':
            logging.info('Initiating the Continuous Greedy algorithm using Polynomial Estimator...')
            y, track, bases = newProblem.polynomial_continuous_greedy(args.center, args.degree, int(args.iterations))
            sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
            output += "_degree_" + str(args.degree) + "_around_" + str(args.center)

        if args.estimator == 'sampler':
            logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator...')
            y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations)
            output += "_" + str(args.samples) + "_samples"
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
            for key in track:
                results.append((key, multilinear_relaxation(newProblem.utility_function, track[key][1])))
        else:
            # results = [(args.constraints, track[args.iterations - 1][0], newProblem.utility_function(y))]
            # results = [(args.samples, y, newProblem.utility_function(y))]
            results = []
            for key in track:
                results.append((key, multilinear_relaxation(newProblem.utility_function, track[key][1])))
        save(output, results)

    else:
        # y = dict.fromkeys(newProblem.groundSet, 0.5)
        if os.path.exists("random_y"):
            y = load("random_y")
        else:
            y = dict(zip(newProblem.groundSet, np.random.rand(newProblem.problemSize).tolist()))
            print(y)
            save("random_y", y)

        out = multilinear_relaxation(y)
        sys.stderr.write("multilinear relaxation is: " + str() + '\n')
        if args.estimator == 'polynomial':
            output = directory_output + args.problemType + "_" + args.input.split("/")[-1] + "_" + args.estimator \
                                      + "_" + str(args.center) + "_y_random"
            start = time()
            poly_grad, poly_estimation = newProblem.get_polynomial_estimator(args.center, args.degree)\
                .estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated grad is: " + str(poly_grad) + '\n')
            sys.stderr.write("estimated value of the function is: " + str(poly_estimation) + '\n')
            if os.path.exists(output):
                poly_results = load(output)
                poly_results.append((elapsed_time, args.degree, poly_estimation, out))
            else:
                poly_results = [(elapsed_time, args.degree, poly_estimation, out)]
            save(output, poly_results)

        if args.estimator == 'sampler':
            output = directory_output + args.problemType + "_" + args.input.split("/")[-1] + "_" + args.estimator \
                                      + "_y_random"
            start = time()
            sampler_grad, sampler_estimation = newProblem.get_sampler_estimator(args.samples)\
                                                         .estimate(y)
            elapsed_time = time() - start
            sys.stderr.write("estimated value of the function is: " + str(sampler_estimation) + '\n')
            if os.path.exists(output):
                sampler_results = load(output)
                sampler_results.append((elapsed_time, args.samples, sampler_estimation, out))
            else:
                sampler_results = [(elapsed_time, args.samples, sampler_estimation, out)]
            save(output, sampler_results)

        if args.estimator == 'samplerWithDependencies':
            sampler_output = directory_output + args.problemType + '_1_graph_y_rand2' + '_samp_with_dep_estimation'
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
