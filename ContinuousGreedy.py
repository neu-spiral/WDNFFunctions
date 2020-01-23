import numpy as np
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
    samples = dict.fromkeys(y.iterkeys(), 0.0)
    p = dict.fromkeys(y.iterkeys(), np.random.rand())
    if dependencies != {}:
        for element in dependencies.keys():
            if y[element] > p[element]:
                samples[element] = 1
    else:
        for i in y.keys():
            if y[i] > p[i]:
                samples[i] = 1
    return samples




class GradientEstimator(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different gradient estimators.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3


    @abstractmethod
    def __init__(self):
        """
        """
        pass


    def estimate(self, y):
        pass




class SamplerEstimator(GradientEstimator):
    """
    """


    def __init__(self, func, numOfSamples):
        """func is either log or queueSize.
        """
        self.func = func
        self.numOfSamples = numOfSamples


    def estimate(self, y):
        """y is a dictionary of {item: value} pairs.
        """
        grad = dict.fromkeys(y.iterkeys(), 0.0)
        for j in range(self.numOfSamples):
            x = generateSamples(y).copy()
            for i in y.keys():
                x1 = x.copy()
                x1[i] = 1
                x0 = x.copy()
                x0[i] = 0
                grad[i] += self.func(x1) - self.func(x0)
        grad = {key: grad[key] / self.numOfSamples for key in grad.keys()}
        return grad




class SamplerEstimatorWithDependencies(GradientEstimator):
    """
    """


    def __init__(self, dependencies, func, numOfSamples):
        """func is either log or queueSize.
        """
        self.dependencies = dependencies
        self.func = func
        self.numOfSamples = numOfSamples


    def estimate(self, y):
        grad = dict.fromkeys(y.iterkeys(), 0.0)
        for j in range(self.numOfSamples):
            x = generateSamples(y, self.dependencies)
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


    def __init__(self, my_wdnf):
        """my_wdnf is a wdnf object
        """
        self.my_wdnf = my_wdnf


    def estimate(self, y):
        grad = dict.fromkeys(y.iterkeys(), 0.0)
        for key in self.my_wdnf.findDependencies().keys():
            y1 = y.copy()
            y1[key] = 1
            grad1 = self.my_wdnf(y1)

            y0 = y.copy()
            y0[key] = 0
            grad0 = self.my_wdnf(y0)

            delta = grad1 - grad0
            grad[key] += delta
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




class UniformMatroidSolver(LinearSolver):
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
        """partitionedSet is a dictionary of sets, k_list is a
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
            for key in self.partitionedSet[partition]:
                filtered_gradient = {key: gradient[key] for key in self.partitionedSet[partition]}
            selection = UniformSolver.solve(filtered_gradient)
            result[partition] = selection
        return result




class ContinuousGreedy():
    """
    """


    def __init__(self, linearSolver, estimator, initialPoint):
        """
        """
        self.linearSolver = linearSolver
        self.estimator = estimator
        self.initialPoint = initialPoint


    def FW(self, iterations):
        x0 = self.initialPoint.copy()
        gamma = 1.0 / iterations
        y = x0.copy()
        for t in range(iterations):
            #print(t)
            gradient = self.estimator.estimate(y)
            #print(gradient)
            mk = self.linearSolver.solve(gradient) #finds maximum
            indices = set()
            for value in mk.values(): #updates y
                indices = indices.union(value)
            for i in list(indices):
                y[i] = y[i] + gamma
            #print(y)
        return y
