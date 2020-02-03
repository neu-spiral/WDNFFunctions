from ContinuousGreedy import LinearSolver, PartitionMatroidSolver, SamplerEstimator, PolynomialEstimator, ContinuousGreedy
from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation, log
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate a random rewards dataset',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problemType', default = 'DR', help = 'Type of the problem instance', choices = ['DR', 'QS', 'FL', 'IM'])
    parser.add_argument('--rewardsInput', default = "rewards.txt", help = 'Input file that stores rewards')
    parser.add_argument('--partitionsInput', default = "givenPartitions.txt", help = 'Input file that stores partitions')
    parser.add_argument('--typesInput', default = "types.txt", help = 'Input file that stores targeted partitions of the ground set')
    parser.add_argument('--constraints', default = {1: 15, 2: 42}, help = 'Constraints dictionary with {type:cardinality} pairs')
    parser.add_argument('--estimator', default = "sampler", help = 'Type of the estimator', choices = ['sampler', 'polynomial', 'samplerWithDependencies'])
    parser.add_argument('--iterations', default = 100, help = 'Number of iterations used in the Frank-Wolfe algorithm')
    parser.add_argument('--degree', default = 8, help = 'Degree of the polynomial estimator')
    parser.add_argument('--center', default = 0.0, help = 'The point around which Taylor approximation is calculated')
    parser.add_argument('--samples', default = 100, help = 'Number of samples used to calculate the sampler estimator')
    parser.add_argument('--timeOutput', default = "sampler_time.txt", help = 'File in which time of each iteration is stored')
    parser.add_argument('--objectiveOutput', default = "sampler_obj.txt", help = 'File in which objective at each iteration is stored')
    args = parser.parse_args()

    #rewards = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.6, 5: 0.5, 6: 0.4} #{x_i: r_i} pairs
    #givenPartitions = {'fruits': (1, 5), 'things': (2, 3), 'actions': (4, 6)} #{P_i: (x_j)} pairs where x_j in P_i
    #types = {1: 'noun', 2: 'noun', 3: 'noun', 4: 'verb', 5: 'noun', 6: 'verb'} #{x_i: type} pairs
    #k_list = {'verb': 1, 'noun': 2}

    rewards = eval(open(args.rewardsInput, 'r').read())
    givenPartitions = eval(open(args.partitionsInput, 'r').read())
    types = eval(open(args.typesInput, 'r').read())
    k_list = args.constraints

    if args.problemType == 'DR':
        newProblem = DiversityReward(rewards, givenPartitions, log, types, k_list)

    if args.estimator == 'polynomial':
        Y1, track1, bases1 = newProblem.PolynomialContinuousGreedy(args.center, args.degree, args.iterations)
        objective = 0.0
        time_list = []
        obj_list = []
        for i in track1:
            for wdnf_instance in newProblem.wdnf_list:
                objective += np.log1p(wdnf_instance(i[1]))
            time_list.append(i[0])
            obj_list.append(objective)
         #print('(Polynomial) Time elapsed: ' + str(i[0]) + '    Objective is: ' + str(objective) + '   Gradient is: ' + str(i[2]))

    if args.estimator == 'sampler':
        Y2, track2, bases2 = newProblem.SamplerContinuousGreedy(args.samples, args.iterations)
        objective = 0.0
        time_list = []
        obj_list = []
        for j in track2:
            for wdnf_instance in newProblem.wdnf_list:
                objective += np.log1p(wdnf_instance(j[1]))
            time_list.append(j[0])
            obj_list.append(objective)

    f = open(args.timeOutput, "w")
    f.write(str(time_list))
    f.close()

    f = open(args.objectiveOutput, "w")
    f.write(str(obj_list))
    f.close()
        #print('(Sampler) Time elapsed: ' + str(j[0]) + '    Objective is: ' + str(objective) + '   Gradient is:  ' + str(j[2]))

    #plt.plot(time_list1, obj_list1, 'r^', time_list2, obj_list2, 'g^')
    #plt.show()



##Test segment for testing the gradients for identity function starts
    #newSamplerEstimator = newProblem.getSamplerEstimator(100)
    #newPolynomialEstimator = newProblem.getPolynomialEstimator(0, 4)

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
