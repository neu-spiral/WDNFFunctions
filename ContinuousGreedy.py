# # demands is list of object instances, edges is dictionary with (u,v) as key and mu_uv as value
import numpy as np
import math
#import itertools
from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from heapq import nlargest
from wdnf import wdnf, poly, taylor
# from time import time
# from helpers import write
# import argparse
# import os


def generateSamples(y, dependencies = {}):
    """ Generates random samples x for e in dependencies P(x_e = 1) = y_e
    """
    samples = [0.0] * len(y)
    p = np.random.rand(len(y))
    if dependencies != {}:
        for element in dependencies.keys():
            if y[element] > p[element]:
                samples[element] = 1
    else:
        for i in range(len(y)):
            if y[i] > p[i]:
                samples[i] = 1
    return samples


def derive(type, x, degree):
    if type == 'ln':
        if degree == 0:
            return math.log1p(x) #log1p(x) is ln(x+1)
        else:
            return (((-1.0)**degree) * math.factorial(degree)) / ((1.0 + x)**(degree + 1))
    if type == 'queueSize':
        if degree == 0:
            return 1.0 / (1.0 - x)
        else:
            return math.factorial(degree) / ((1.0 - x)**(degree + 1))


def findDerivatives(type, center, degree):
    derivatives = []
    for i in range(degree + 1):
        derivatives.append(derive(type, center, i))
    return derivatives


class GradientEstimator(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different gradient estimators.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3


    @abstractmethod
    def __init__(self):
        """
        """
        pass


    def estimate(self, y): #Should the estimate's take y as an input?
        pass


class SamplerEstimator(GradientEstimator):
    """
    """


    def __init__(self, func, numOfSamples):
        """func is a function of where func(my_wdnf) = log(my_wdnf) or
        func(my_wdnf) = my_wdnf/(1 - my_wdnf)
        """
        self.func = func
        self.numOfSamples = numOfSamples


    def estimate(self, y):
        grad = [0.0] * len(y)
        x = generateSamples(y)
        for i in range(self.numOfSamples):
            x1 = x
            x1[i] = 1
            x0 = x
            x0[i] = 0
            grad[i] += self.func(x1) - self.func(x0)
        grad = grad / self.numOfSamples
        return grad


class SamplerEstimatorWithDependencies(GradientEstimator): #
    """
    """


    def __init__(self, my_wdnf, func, numOfSamples):
        """my_wdnf is a wdnf object and func is a function of that wdnf object
        such as func(my_wdnf) = log(my_wdnf) or
        func(my_wdnf) = my_wdnf/(1 - my_wdnf)
        """
        self.my_wdnf = my_wdnf
        self.func = func
        self.numOfSamples = numOfSamples


    def estimate(self, y):
        grad = [0.0] * len(y)
        x = generateSamples(y, self.my_wdnf.findDependencies())
        for i in range(self.numOfSamples):
            x1 = x
            x1[i] = 1
            x0 = x
            x0[i] = 0
            grad[i] += self.func(x1) - self.func(x0)
        grad = grad / self.numOfSamples
        return grad


class PolynomialEstimator(GradientEstimator):
    """
    """


    def __init__(self, my_wdnf, degree):
        """my_wdnf is a wdnf object
        """
        self.my_wdnf = my_wdnf
        self.degree = degree


    def estimate(self, y):
        grad = [0.0] * len(y)
        for key in self.my_wdnf.findDependencies().keys():
            y1 = y
            y1[key] = 1
            grad1 = self.my_wdnf(y1)

            y0 = y
            y0[key] = 0
            grad0 = self.my_wdnf(y0)

            delta = grad1 - grad0
            grad += delta
        return grad


class LinearSolver(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent solver classes with different constraints.
    """
    __metaclass__ = ABCMeta #Comment this line for Python 3


    @abstractmethod
    def __init__(self, sets, constraints):
        pass


    def solve(self, gradient):
        """Abstract method to solve submodular maximization problems
        according to given constraints where gradient is a GradientEstimator
        object.
        """
        pass


class UniformMatroidSolver(LinearSolver): #tested, works
    """
    """


    def __init__(self, groundSet, k):
        """
        """
        self.groundSet = groundSet
        self.k = k


    def solve(self, gradient):
        """
        """
        return set(nlargest(self.k, gradient, key = gradient.get))


class PartitionMatroidSolver(LinearSolver): #tested for distinct partititons, works -should be revised for overlapping partitions
    """
    """


    def __init__(self, partitionedSet, k_list):
        """Partitioned set is a dictionary of dictionaries, k_list is a
        dictionary of cardinalities.
        """
        self.partitionedSet = partitionedSet
        self.k_list = k_list


    def solve(self, gradient):
        """
        """
        result = {}
        selection = []
        for partition in self.partitionedSet:
            UniformSolver = UniformMatroidSolver(self.partitionedSet[partition], self.k_list[partition])
            filtered_gradient = {key: gradient[key] for key in self.partitionedSet[partition]}
            selection = UniformSolver.solve(filtered_gradient)
            result[partition] = selection
        return result


class ContinuousGreedy():
    """
    """


    def __init__(linearSolver, estimator):
        """
        """
        self.linearSolver = linearSolver
        self.estimator = estimator


    def FW(self, iterations):
        x0 = [0.0] * len(self.y) ##How to keep the size information?
        gamma = 1.0 / iterations
        y = x0
        for t in range(iterations):
            gradient = self.estimator.estimate(y)
            mk = (self.linearSolver).solve(gradient) #finds maximum
            for i in mk: #updates y
                y[i] += gamma
        return y


if __name__ == "__main__":
    actors = {'act1', 'act2', 'act3', 'act4', 'act5'}
    actors_gradient = {'act1': 1000, 'act2': 300, 'act3': 400, 'act4': 500, 'act5': 700}
    NewUniSolver = UniformMatroidSolver(actors, 3)
    print(NewUniSolver.solve(actors_gradient))
    print(isinstance(NewUniSolver, UniformMatroidSolver))

    directors = {'dir1', 'dir2', 'dir3'}
    directors_gradient = {'dir1': 1500, 'dir2': 1200, 'dir3': 250}
    figurants = {'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7'}
    figurants_gradient = {'fig1': 10, 'fig2': 20, 'fig3': 35, 'fig4': 5, 'fig5': 6, 'fig6': 2, 'fig7': 13}
    candidates = {'actors': actors, 'directors': directors, 'figurants': figurants}
    candidates_gradient = {}
    candidates_gradient.update(actors_gradient)
    candidates_gradient.update(directors_gradient)
    candidates_gradient.update(figurants_gradient)
    k_list = {'actors': 2, 'directors': 1, 'figurants': 5}
    NewPartSolver = PartitionMatroidSolver(candidates, k_list)
    print(NewPartSolver.solve(candidates_gradient))
    print(findDerivatives('queueSize', 3, 5))
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
