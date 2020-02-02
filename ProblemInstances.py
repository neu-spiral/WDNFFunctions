from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from ContinuousGreedy import LinearSolver, UniformMatroidSolver, PartitionMatroidSolver, SamplerEstimator, PolynomialEstimator, ContinuousGreedy
from networkx import Graph, DiGraph
from time import time
from wdnf import wdnf, poly, taylor
import argparse
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt




def log(wdnf_list, x):
    """ Given a list of wdnf objects and a vector x as a dictionary, returns the
    f(x) = sum_{i=1}^K log(sum_j r_j x_{j} + 1)
    """
    output = 0.0
    for wdnf_object in wdnf_list:
        output += wdnf_object.evaluate(x, np.log1p)
    return output


def qs(x):
    """Given rho returns rho / (1 - rho)
    """
    return x / (1.0 - x)


def queueSize(wdnf_list, x):
    """ Given a list of wdnf objects and a vector x as a dictionary, returns the
    f(x) = sum_{i=1}^K qs(sum_j r_j x_{j} + 1)
    """
    output = 0.0
    for wdnf_object in wdnf_list:
        output += wdnf_object.evaluate(x, qs)
    return output


def derive(type, x, degree):
    """Helper function to create derivatives list of taylor objects. Given the
    degree and the center of the taylor expansion with the type of the functions
    returns the value of the function's derivative at the given center point.
    """
    if type == log:
        if degree == 0:
            return np.log1p(x) #log1p(x) is ln(x+1)
        else:
            return (((-1.0)**degree) * math.factorial(degree - 1)) / ((1.0 + x)**(degree))
    if type == qs:
        if degree == 0:
            return qs(x)
        else:
            return math.factorial(degree) / ((1.0 - x)**(degree + 1))
    if type == id:
        if degree == 0:
            return x
        elif degree == 1:
            return 1
        else:
            return 0


def findDerivatives(type, center, degree):
    """Type is either 'ln' or 'queueSize', helper function to create the
    derivatives list of taylor objects.
    """
    derivatives = []
    for i in range(degree + 1):
        derivatives.append(derive(type, center, i))
    return derivatives


def evaluateAll(taylor_instance, wdnf_list):
    my_wdnf = wdnf(dict(), wdnf_list[0].sign)
    #print(my_wdnf.coefficients)
    for wdnf_instance in wdnf_list:
        #print(wdnf_instance.coefficients)
        #print(taylor_instance.compose(wdnf_instance).coefficients)
        my_wdnf += taylor_instance.compose(wdnf_instance)
    return my_wdnf


class Problem(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different problem instances.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3


    @abstractmethod
    def __init__(self):
        """
        """
        pass


    def getSolver(self):
        """
        """
        pass


    def getSamplerEstimator(self, numOfSamples):
        """
        """
        pass


    def getPolynomialEstimator(self, center, degree):
        """
        """
        pass


    def getInitialPoint(self):
        """
        """
        pass


    def SamplerContinuousGreedy(self, numOfSamples, iterations):
        """
        """
        newCG = ContinuousGreedy(self.getSolver(), self.getSamplerEstimator(numOfSamples), self.getInitialPoint())
        return newCG.FW(iterations, True)


    def PolynomialContinuousGreedy(self, center, degree, iterations):
        """
        """
        newCG = ContinuousGreedy(self.getSolver(), self.getPolynomialEstimator(center, degree), self.getInitialPoint())
        return newCG.FW(iterations, True)




class DiversityReward(Problem):
    """
    """


    def __init__(self, rewards, givenPartitions, fun, types, k_list):
        """ rewards is a dictionary containing {word: reward} pairs,
        givenPartitions is a dictionary containing {partition: word tuples}, fun
        is either log or queueSize, types is a dictionary containing {word: type} pairs,
        k_list is a dictionary of {type: cardinality} pairs.
        """
        wdnf_list = []
        partitionedSet = {}
        for i in givenPartitions:
            coefficients = {}
            for j in givenPartitions[i]:
                coefficients[j] = rewards[j]
                if partitionedSet.has_key(types[j]):
                    partitionedSet[types[j]].add(j)
                else:
                    partitionedSet[types[j]] = {j}
            new_wdnf = wdnf(coefficients, 1)
            wdnf_list.append(new_wdnf)
        self.rewards = rewards
        self.wdnf_list = wdnf_list
        #for i in self.wdnf_list:
            #print(i.coefficients)
        self.partitionedSet = partitionedSet
        self.fun = fun
        self.k_list = k_list
        #self.problemSize = len(rewards)


    def getSolver(self):
        """
        """
        return PartitionMatroidSolver(self.partitionedSet, self.k_list)


    def func(self, x):
        """
        """
        return self.fun(self.wdnf_list, x)


    def getSamplerEstimator(self, numOfSamples):
        """
        """
        return SamplerEstimator(self.func, numOfSamples)


    def getPolynomialEstimator(self, center, degree):
        """
        """
        derivatives = findDerivatives(self.fun, center, degree)
        #print('derivatives are: ' + str(derivatives))
        myTaylor = taylor(degree, derivatives, center)
        #print('coefficients of the Taylor expansion are:' + str(myTaylor.poly_coef))
        #print('degree of the Taylor expansion is: ' + str(myTaylor.degree))
        #derivatives = [1] + [0] * (degree - 1)
        #myTaylor = taylor(degree, [1, 0)
        my_wdnf = evaluateAll(myTaylor, self.wdnf_list)
        print('my_wdnf:' + str(my_wdnf.coefficients))
        #print(my_wdnf(y))
        return PolynomialEstimator(my_wdnf)


    def getInitialPoint(self):
        """
        """
        return dict.fromkeys(self.rewards.iterkeys(), 0.0)




class QueueSize(Problem):
    """
    """


    def __init__(self):
        """
        """
        pass


    def getSolver(self):
        """
        """
        pass


    def getSamplerEstimator(self, numOfSamples):
        """
        """
        pass


    def getPolynomialEstimator(self, center, degree):
        """
        """
        pass


    def getInitialPoint(self):
        """
        """
        pass




class InfluenceMaximization(Problem):
    """
    """


    def __init__(self, graph, fun, constraints, targetPartitions = None):
        """ graph is a Graph object from networkx. If given, targetPartitions is a dictionary with {node : type}
        pairs and converts the problem to an Influence Maximization over partition matroids, fun is log, constraints is an
        integer denoting the number of seeds if constraints is over uniform matroid or a dictionary with {type : int} pairs
        if over partition matroids.
        """

        self.groundSet = graph.nodes()
        self.edges = graph.edges()
        self.fun = fun
        self.constraints = constraints
        self.targetPartitions = targetPartitions

        givenPartitions = dict()
        wdnf_list = []
        paths = dict(nx.all_pairs_shortest_path(graph))
        for node1 in self.groundSet: ##this is not efficient. More efficient way?
            givenPartitions[node1] = ()
            #groundSet[node1] = paths[node1].keys()
            for node2 in self.groundSet:
                if nx.has_path(graph, node2, node1):
                    givenPartitions[node1] += (node2,)
            wdnf_list.append(wdnf({givenPartitions[node1]: 1}, -1))
        self.givenPartitions = givenPartitions.copy()
        self.wdnf_list = wdnf_list
        #self.groundSet = groundSet.copy()


    def getSolver(self):
        """
        """
        if self.targetPartitions == None:
            solver = UniformMatroidSolver(self.groundSet, self.constraints)
        else:
            solver  = PartitionMatroidSolver(self.targetPartitions, self.constraints)
        return solver


    #def


    def getSamplerEstimator(self, numOfSamples):
        """
        """
        pass


    def getPolynomialEstimator(self, center, degree):
        """
        """
        pass


    def getInitialPoint(self):
        """
        """
        pass




class FacilityLocation(Problem):
    """
    """


    def __init__(self):
        """
        """
        pass


    def getSolver(self):
        """
        """
        pass


    def getSamplerEstimator(self, numOfSamples):
        """
        """
        pass


    def getPolynomialEstimator(self, center, degree):
        """
        """
        pass


    def getInitialPoint(self):
        """
        """
        pass




if __name__ == "__main__":

    graph = DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6])
    graph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5), (4, 6), (6, 3)])
    newProblem = InfluenceMaximization(graph, log, 5)
    for i in newProblem.givenPartitions.keys():
        print(newProblem.givenPartitions[i].coefficients)
    #print(newProblem.groundSet)


    # parser = argparse.ArgumentParser(description = 'Run the Continuous Greedy Algorithm', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('output')
    # parser.add_argument('--iterations', help = 'Number of iterations in the Frank-Wolfe Algorithm')
    # parser.add_argument('--numOfSamples', help = 'Number of samples generated by the Sampler Estimator')
    # parser.add_argument('--degree', help = 'Order of the Taylor expansion used by the Polynomial Estimator')
    #
    #
    #
    #
    # rewards = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.6, 5: 0.5, 6: 0.4} #{x_i: r_i} pairs
    # givenPartitions = {'fruits': (1, 5), 'things': (2, 3), 'actions': (4, 6)} #{P_i: (x_j)} pairs where x_j in P_i
    # types = {1: 'noun', 2: 'noun', 3: 'noun', 4: 'verb', 5: 'noun', 6: 'verb'} #{x_i: type} pairs
    # k_list = {'verb': 1, 'noun': 2}
    # newProblem = DiversityReward(rewards, givenPartitions, id, types, k_list)

    # Y1, track1, bases1 = newProblem.PolynomialContinuousGreedy(0, 4, 150)
    # objective = 0.0
    # time_list1 = []
    # obj_list1 = []
    # for i in track1:
    #     for wdnf_instance in newProblem.wdnf_list:
    #         objective += np.log1p(wdnf_instance(i[1]))
    #     time_list1.append(i[0])
    #     obj_list1.append(objective)
    #     print('(Polynomial) Time elapsed: ' + str(i[0]) + '    Objective is: ' + str(objective) + '   Gradient is: ' + str(i[2]))
    #
    #
    # Y2, track2, bases2 = newProblem.SamplerContinuousGreedy(200, 100)
    # objective = 0.0
    # time_list2 = []
    # obj_list2 = []
    # for j in track2:
    #     for wdnf_instance in newProblem.wdnf_list:
    #         objective += np.log1p(wdnf_instance(j[1]))
    #     time_list2.append(j[0])
    #     obj_list2.append(objective)
    #     print('(Sampler) Time elapsed: ' + str(j[0]) + '    Objective is: ' + str(objective) + '   Gradient is:  ' + str(j[2]))

    #plt.plot(time_list1, obj_list1, 'r^', time_list2, obj_list2, 'g^')
    #plt.show()



    # actors = {'act1', 'act2', 'act3', 'act4', 'act5'}
    # actors_gradient = {'act1': 1000, 'act2': 300, 'act3': 400, 'act4': 500, 'act5': 700}
    # NewUniSolver = UniformMatroidSolver(actors, 3)
    # print(NewUniSolver.solve(actors_gradient))
    # print(isinstance(NewUniSolver, UniformMatroidSolver))
    #
    # directors = {'dir1', 'dir2', 'dir3'}
    # directors_gradient = {'dir1': 1500, 'dir2': 1200, 'dir3': 250}
    # figurants = {'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7'}
    # figurants_gradient = {'fig1': 10, 'fig2': 20, 'fig3': 35, 'fig4': 5, 'fig5': 6, 'fig6': 2, 'fig7': 13}
    # candidates = {'actors': actors, 'directors': directors, 'figurants': figurants}
    # candidates_gradient = {}
    # candidates_gradient.update(actors_gradient)
    # candidates_gradient.update(directors_gradient)
    # candidates_gradient.update(figurants_gradient)
    # k_list = {'actors': 2, 'directors': 1, 'figurants': 5}
    # NewPartSolver = PartitionMatroidSolver(candidates, k_list)
    # print(NewPartSolver.solve(candidates_gradient))
    # print(findDerivatives('queueSize', 3, 5))
    #
    # kids = {'kid1': {'goalkeeper': 100, 'defender': 50, 'forward': 25},
    #         'kid2': {'goalkeeper': 80, 'defender': 150, 'forward': 30},
    #         'kid3': {'goalkeeper': 10, 'defender': 35, 'forward': 250},
    #         'kid4': {'goalkeeper': 300, 'defender': 5, 'forward': 125},
    #         'kid5': {'goalkeeper': 20, 'defender': 50, 'forward': 75},
    #         'kid6': {'goalkeeper': 50, 'defender': 28, 'forward': 36},
    #         'kid7': {'goalkeeper': 60, 'defender': 90, 'forward': 12},
    #         'kid8': {'goalkeeper': 70, 'defender': 90, 'forward': 450},
    #         'kid9': {'goalkeeper': 45, 'defender': 350, 'forward': 30},
    #         'kid10': {'goalkeeper': 40, 'defender': 48, 'forward': 45},
    #         'kid11': {'goalkeeper': 175, 'defender': 12, 'forward': 120}}
    #
    # def findPartition(items, partition):
    #     result = {}
    #     for item in items:
    #         result[item] = items[item][partition]
    #     return result
    #
    # goalkeepers = findPartition(kids, 'goalkeeper')
    # defenders = findPartition(kids, 'defender')
    # forwards = findPartition(kids, 'forward')
    #
    # players = {'goalkeepers': goalkeepers, 'defenders': defenders, 'forwards': forwards}
    # player_list = {'goalkeepers': 1, 'defenders': 5, 'forwards': 5}
    # print(NewPartSolver.solve(players, player_list))
