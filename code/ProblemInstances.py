from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from ContinuousGreedy import LinearSolver, UniformMatroidSolver, PartitionMatroidSolver, SamplerEstimator, PolynomialEstimator, ContinuousGreedy
from networkx import Graph, DiGraph
from networkx.algorithms import bipartite
from time import time
from wdnf import WDNF, poly, taylor
import argparse
import math
import networkx as nx
import numpy as np
import sys
#import matplotlib.pyplot as plt




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
    """Type is either 'log' or 'queueSize', helper function to create the
    derivatives list of taylor objects.
    """
    derivatives = []
    for i in range(degree + 1):
        derivatives.append(derive(type, center, i))
    return derivatives


def evaluateAll(taylor_instance, wdnf_list):
    my_wdnf = WDNF(dict(), wdnf_list[0].sign)
    for wdnf_instance in wdnf_list:
        my_wdnf += taylor_instance.compose(wdnf_instance)
    return my_wdnf


def ro_uv(edges, demands, x):
    """ edges is a list of edges in (u, v) form where (u, v) is an edge from u to v.
        demands is a list of Demand objects
    """
    ro_uv = {}
        #Initialize the functions...
    for edge in edge_dict:
        ro_uv[edge] = 0.0
        # Go through demands
    for demand in demands:
        path = demand['path']
        item = demand['item']
        rate = demand['rate']
        if  x[(path[0], item)] == 0 and len(path) > 1:
            for node_i in range(len(path) - 1):
                edge = (path[node_i], path[node_i + 1])
                ro_uv[edge] = ro_uv[edge] + rate
                if x[(path[node_i + 1], item)] == 1:
                     break
    return ro_uv   #is a dictionary of {(u, v): load}


class Demand:
    """ A demand object. Contains the item requested, the path a request follows, as a list, and the
        rate with which requests are generated. Tallies count various metrics.

        Attributes:
        item: the id of the item requested
        path: a list of nodes to be visited
        rate: the rate with which this request is generated
        query_source: first node on the path
        item_source: last node on the path
    """

    def __init__(self, item, path, rate):
        """ Initialize a new request.
        """
        self.item = item
        self.path = path
        self.rate = rate

        self.query_source = path[0]
        self.item_source = path[-1]

    def __str__(self):
        return Demand.__repr__(self)

    def __repr__(self):
        return 'Demand(' + ','.join(map(str, [self.item, self.path, self.rate])) + ')'

    def succ(self, node):
        """ The successor of a node in the path.
        """
        path = self.path
        if node not in path:
            return None
        i = path.index(node)
        if i + 1 == len(path):
            return None
        else:
            return path[i + 1]

    def pred(self, node):
        """The predecessor of a node in the path.
        """
        path = self.path
        if node not in path:
            return None
        i = path.index(node)
        if i - 1 < 0:
            return None
        else:
            return path[i - 1]


class Problem(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different problem instances.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3

    @abstractmethod
    def __init__(self):
        """
        """
        self.problemSize = 0
        self.groundSet = set()
        self.utility_function()

    def utility_function(self, y={}):
        pass

    def getSolver(self):
        """
        """
        pass

    def func(self):
        """
        """
        pass

    def getSamplerEstimator(self, numOfSamples):
        """
        """
        return SamplerEstimator(self.utility_function, numOfSamples)

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
        return newCG.fw(iterations, False)

    def PolynomialContinuousGreedy(self, center, degree, iterations):
        """
        """
        newCG = ContinuousGreedy(self.getSolver(), self.getPolynomialEstimator(center, degree), self.getInitialPoint())
        return newCG.fw(iterations, False)


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
        self.partitionedSet = partitionedSet
        self.fun = fun
        self.k_list = k_list
        #self.utility_function()
        self.problem_size = len(rewards) ####revise


    def utility_function(self):
        pass


    def getSolver(self):
        """
        """
        return PartitionMatroidSolver(self.partitionedSet, self.k_list)


    def func(self, x):
        """
        """
        return self.fun(self.wdnf_list, x)


    def getPolynomialEstimator(self, center, degree):
        """
        """
        derivatives = findDerivatives(self.fun, center, degree)
        myTaylor = taylor(degree, derivatives, center)
        my_wdnf = evaluateAll(myTaylor, self.wdnf_list)
        print('my_wdnf:' + str(my_wdnf.coefficients))
        return PolynomialEstimator(my_wdnf)


    def getInitialPoint(self):
        """
        """
        return dict.fromkeys(self.rewards.iterkeys(), 0.0)




class QueueSize(Problem):
    """
    """


    def __init__(self, graph, capacities, demands):
        """ graph is a symmetrical directed graph,
        capacities is a dictionary with {node: capacity} pairs,
        demands is a list of demand objects
        """
        pass


    def utility_function(self):
        pass


    def getSolver(self):
        """
        """
        pass


    def func(self, x):
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


    def __init__(self, graphs, constraints, targetPartitions = None):
        """ graphs is a list of DiGraph objects from networkx. If given, targetPartitions is a dictionary with {node : type}
        pairs and converts the problem to an Influence Maximization over partition matroids, constraints is an
        integer denoting the number of seeds if constraints is over uniform matroid or a dictionary with {type : int} pairs
        if over partition matroids.
        """
        #self.graphs = graphs
        self.groundSet = set(graphs[0].nodes()) #all graphs share the same set of nodes
        self.problemSize = graphs[0].number_of_nodes() #|V|
        self.instancesSize = len(graphs) #|G|
        self.constraints = constraints #number of seeds aka k
        self.targetPartitions = targetPartitions
        #givenPartitions = dict()
        wdnf_dict = dict()
        for i in range(self.instancesSize):
            self.edges = graphs[i].edges() #edges are different for each graph
            P = dict()
            wdnf_list = dict()
            dependencies = set()
            paths = nx.algorithms.dag.transitive_closure(graphs[i])
            wdnfSoFar = WDNF(dict(), -1)
            for node in self.groundSet:
                P[node] = tuple(sorted([node] + list(paths.predecessors(node)))) # P is {v, P_v} pairs
                #wdnf_list[node1] = WDNF({P[node1]: 1}, -1)
                wdnfSoFar += (1.0 / self.problemSize) * (WDNF({(): 1.0}, -1) + ((-1.0) * WDNF({P[node]: 1.0}, -1)))
                #sys.stderr.write("wndfSoFar is: " + str(wdnfSoFar.coefficients))
            #givenPartitions[i] = P.copy() #givenPartitions is a dictionary of (v: P_v) pairs where v is a node in graph and P_v is the set of all nodes having a (directed) path to v (in tuple format)
            wdnf_dict[i] = wdnfSoFar #prod(1 - x_u) for all u in P_v
            dependencies = dependencies.union(wdnfSoFar.find_dependencies())
        self.wdnf_dict = wdnf_dict
        #self.utility_function()
        self.dependencies = dependencies


    def utility_function(self, y):
        """
        :param y:
        :return:
        """
        objective = [(1.0 / self.instancesSize) * np.log1p(self.wdnf_dict[graph](y)) for graph in range(self.instancesSize)]
        return sum(objective)


    def getSolver(self):
        """
        """
        if self.targetPartitions == None:
            solver = UniformMatroidSolver(self.groundSet, self.constraints)
        else:
            solver  = PartitionMatroidSolver(self.targetPartitions, self.constraints)
        return solver


    #def func(self, x):
    #    output = 0.0
    #    for i in range(self.instancesSize):
    #        sum = 0.0
    #        for node in self.groundSet:
    #            sum += 1.0 - self.wdnf_dict[i][node](x)
    #        output += np.log1p(sum / self.problemSize)
    #    return output / (self.instancesSize * 1.0)


    def getPolynomialEstimator(self, center, degree):
        """
        """
        derivatives = findDerivatives(log, center, degree)
        myTaylor = taylor(degree, derivatives, center)
        final_wdnf = WDNF(dict(), -1)
        for i in range(self.instancesSize):
            wdnfSoFar = WDNF(dict(), -1)
            for node in self.groundSet:
                wdnfSoFar += WDNF({(): 1.0}, -1) + ((-1.0) * self.wdnf_dict[i][node]) ###edit here
            my_wdnf = myTaylor.compose((1.0 / self.problemSize) * wdnfSoFar + WDNF({(): 1.0}, -1))
            final_wdnf += my_wdnf
        return PolynomialEstimator((1.0 / self.instancesSize) * final_wdnf)


    def getInitialPoint(self):
        """
        """
        return dict.fromkeys(self.groundSet, 0.0)




class FacilityLocation(Problem):
    """
    """


    def __init__(self, B, constraints):
        """ B is a complete weighted bipartite graph, constraints is an integer which denotes the maximum number of facilities
        """
        self.X = {n for n, d in B.nodes(data = True) if d['bipartite'] == 0} #facilities
        self.Y = set(B) - self.X #customers
        self.constraints = constraints
        self.partitionedSet = dict.fromkeys(self.Y, self.X)
        self.size = len(self.Y) #number of customers
        wdnf_dict = dict()
        for y in self.Y:
            weights = {nodeX : B.get_edge_data(nodeX, y)['weight'] for nodeX in self.X}
            weights[len(self.X) + 1] = 0.0
            wdnfSoFar = WDNF(dict(), -1)
            descending_weights = sorted(weights.values(), reverse = True)
            indices = sorted(range(len(weights.values())), key = lambda k: weights.values()[k], reverse = True)
            for i in range(len(self.X)):
                index = tuple(index + 1 for index in indices[:(i+1)])
                wdnfSoFar += (descending_weights[i] - descending_weights[i + 1]) * (WDNF({(): 1.0}, -1) + (-1.0) * WDNF({index: 1.0}, -1))
            wdnf_dict[y] = wdnfSoFar
        self.wdnf_dict = wdnf_dict


    def utility_function(self):
        pass


    def getSolver(self):
        """
        """
        k_list = dict.fromkeys(self.Y, self.constraints)
        return PartitionMatroidSolver(self.partitionedSet, k_list)


    def func(self, x):
        """
        """
        sum = 0.0
        for y in self.Y:
            sum += np.log1p(self.wdnf_dict[y](x))
        return sum / self.size


    def getPolynomialEstimator(self, center, degree):
        """
        """
        derivatives = findDerivatives(log, center, degree)
        myTaylor = taylor(degree, derivatives, center)
        wdnfSoFar = WDNF(dict(), -1)
        for y in self.Y:
            wdnfSoFar += myTaylor.compose(self.wdnf_dict[y])
        my_wdnf = (1.0 / self.size) * wdnfSoFar
        return PolynomialEstimator(my_wdnf)


    def getInitialPoint(self):
        """
        """
        return dict.fromkeys(self.X, 0.0)




if __name__ == "__main__":

    graph = DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6])
    graph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5), (4, 6), (6, 3)])
    newProblem = InfluenceMaximization([graph], 3)
    Y1, track1, bases1 = newProblem.PolynomialContinuousGreedy(0.0, 5, 100)
    print(Y1)
    #for i in newProblem.givenPartitions.keys():
    #    print(newProblem.givenPartitions[i].coefficients)
    x = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

    #sum = 0.0
    #for node in newProblem.groundSet:
    #    sum += 1 - newProblem.wdnf_list[node - 1](x)
    #print(sum)
    #print(newProblem.groundSet)
    B = Graph()
    B.add_nodes_from([1, 2, 3, 4, 5, 6], bipartite = 0)
    B.add_nodes_from(['a', 'b'], bipartite = 1)
    B.add_weighted_edges_from([(1, 'a', 0.5), (2, 'a', 0.25), (3, 'a', 3.0), (4, 'a', 4.0), (5, 'a', 2.0), (6, 'a', 1.0), (1, 'b', 1.0), (2, 'b', 1.0), (3, 'b', 1.0), (4, 'b', 1.0), (5, 'b', 1.0), (6, 'b', 1.0)])
    #print(B.get_edge_data(1, 'a')['weight'])
    #print(B.edges(data = True))
    x = ['m1', 'm2', 'm3', 'm4']
    #print(x[:4])
    newProb = FacilityLocation(B, 2)
    Y2, track2, bases2 = newProb.PolynomialContinuousGreedy(0.0, 5, 10)
    #print(Y2)




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
