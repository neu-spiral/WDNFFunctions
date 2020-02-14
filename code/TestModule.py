from helpers import save, load
# from networkx import Graph, DiGraph
# from networkx.algorithms import bipartite
# from networkx.convert import to_edgelist
# from networkx.readwrite.edgelist import read_edgelist
from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation
# from wdnf import WDNF, Taylor
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
    parser.add_argument('--input', default='datasets/test_graphs_file', type=str,
                        help='Data input for the InfluenceMaximization problem')
    # parser.add_argument('--cascades', default=1000, type=int,
    #                     help='Number of cascades used in the Independent Cascade model')
    # parser.add_argument('--p', default=0.02, type=float, help='Infection probability')
    # parser.add_argument('--rewardsInput', default="rewards.txt", help='Input file that stores rewards')
    # parser.add_argument('--partitionsInput', default="givenPartitions.txt", help='Input file that stores partitions')
    # parser.add_argument('--typesInput', default="types.txt",
    #                     help='Input file that stores targeted partitions of the ground set')
    parser.add_argument('--constraints', default=4, type=int,
                        help='Constraints dictionary with {type:cardinality} pairs')
    parser.add_argument('--estimator', default='polynomial', type=str, help='Type of the estimator',
                        choices=['sampler', 'polynomial', 'samplerWithDependencies'])
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations used in the Frank-Wolfe algorithm')
    parser.add_argument('--degree', default=2, type=int, help='Degree of the polynomial estimator')
    parser.add_argument('--center', default=0.0, type=float,
                        help='The point around which Taylor approximation is calculated')
    parser.add_argument('--samples', default=30, type=int,
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
        output = directory_output + args.problemType + "test_case" + args.estimator + "_" + str(args.iterations) \
                                  + "_FW"

    if args.estimator == 'polynomial':
        logging.info('Initiating the Continuous Greedy algorithm using Polynomial Estimator...')
        y, track, bases = newProblem.polynomial_continuous_greedy(args.center, args.degree, int(args.iterations))
        sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
        output += "_degree_" + str(args.degree) + "_around_" + str(args.center)

    if args.estimator == 'sampler':
        logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator...')
        y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations)
        output += "_" + str(args.samples) + "samples"
        sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')

    if args.estimator == 'samplerWithDependencies':
        logging.info('Initiating the Continuous Greedy algorithm using Sampler Estimator with Dependencies...')
        y, track, bases = newProblem.sampler_continuous_greedy(args.samples, args.iterations, newProblem.dependencies)
        sys.stderr.write("objective is: " + str(newProblem.utility_function(y)) + '\n')
        output += "_" + str(args.samples) + "samples"

    if os.path.exists(output):
        results = load(output)
        results.append((args.constraints, track[args.iterations - 1][0], newProblem.utility_function(y)))
    else:
        results = [(args.constraints, track[args.iterations - 1][0], newProblem.utility_function(y))]
    save(output, results)


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
