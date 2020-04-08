from abc import ABCMeta, abstractmethod  # ABCMeta works with Python 2, use ABC for Python 3
from heapq import nlargest
from time import time
import logging
import numpy as np
import sys


def generate_samples(y, dependencies={}):
    """
    Generates random samples for the input vector x. Vector elements are binary.
    If dependencies dictionary is non-empty, saves time by only generating
    samples for dependant terms.
    """
    samples = dict.fromkeys(y.iterkeys(), 0.0)
    # p = dict.fromkeys(y.iterkeys(), np.random.rand())
    p = dict(zip(y.iterkeys(), np.random.rand(len(y)).tolist()))
    # sys.stderr.write("p: " + str(p) + "\n")
    # sys.stderr.write("y: " + str(y) + "\n")
    if dependencies != {}:
        # sys.stderr.write("dependencies are: " + str(y))
        indices = [element for element in dependencies.keys() if y[element] > p[element]]
        samples.update(dict.fromkeys(indices, 1.0))
    else:
        indices = [key for key in y.keys() if y[key] > p[key]]
        samples.update(dict.fromkeys(indices, 1.0))
    # sys.stderr.write("samples: " + str(samples) + "\n")
    return samples


def multilinear_relaxation(utility_function, y):
    out = 0.0
    print(y)
    for i in range(2 ** len(y)):
        binary_vector = map(int, list(bin(i)[2:]))
        if len(binary_vector) < len(y):
            binary_vector = [0] * (len(y) - len(binary_vector)) + binary_vector
        x = dict(zip(y.iterkeys(), binary_vector))
        print(x)
        new_term = utility_function(x)
        for key in y:
            if x[key] == 0:
                new_term *= (1.0 - y[key])
            else:
                new_term *= y[key]
        out += new_term
    sys.stderr.write("multilinear relaxation is: " + str(out) + '\n')
    return out


class GradientEstimator(object):  # For Python 3, replace object with ABCMeta
    """
    Abstract class to parent classes of different gradient estimators.
    """
    __metaclass__ = ABCMeta  # Comment out this line for Python 3

    @abstractmethod
    def __init__(self):
        """
        Initializes estimator objects.
        """
        pass

    def estimate(self, y):
        """
        Estimate function to be called in the ContinuousGreedy.
        :param y: a dictionary denoting the fractional vector
        """
        pass


class TrueGradientCalculator(GradientEstimator):
    """

    """

    def __init__(self, utility_function, dependencies={}):
        """

        :param utility_function:
        """
        super(TrueGradientCalculator, self).__init__()
        self.utility_function = utility_function
        self.dependencies = dependencies

    def estimate(self, y):
        """ It calculates the true value of the gradient of the Multilinear Relaxation of the utility function.
        :param y:
        :return:
        """
        grad = dict()
        if self.dependencies != {}:
            for i in self.dependencies.keys():
                y1 = y.copy()
                y1[i] = 1.0

                y0 = y.copy()
                y0[i] = 0.0
                grad[i] = multilinear_relaxation(self.utility_function, y1) \
                    - multilinear_relaxation(self.utility_function, y0)
        else:
            for i in y.keys():
                y1 = y.copy()
                y1[i] = 1.0

                y0 = y.copy()
                y0[i] = 0.0
                grad[i] = multilinear_relaxation(self.utility_function, y1) \
                    - multilinear_relaxation(self.utility_function, y0)

        multilinear_extension = multilinear_relaxation(self.utility_function, y)
        return grad, multilinear_extension


class SamplerEstimator(GradientEstimator):
    """
    Calculates the expectation by evaluating the given function at randomly
    generated samples and taking the average of them.
    """

    def __init__(self, utility_function, num_of_samples, dependencies={}):
        """
        utility_function is the function whose expectation is going to be estimated and
        numOfSamples is an integer. If dependencies is a non-empty dictionary,
        it contains the dependencies as {element : term} pairs.
        """
        super(SamplerEstimator, self).__init__()
        self.utility_function = utility_function
        self.num_of_samples = num_of_samples
        self.dependencies = dependencies

    def estimate(self, y):
        """
        Estimates the function in self using sampling. y is a dictionary of
        {item: value} pairs representing the fractional vector.
        """
        grad = dict.fromkeys(y.iterkeys(), 0.0)
        estimation = 0.0  # this variable will be used for testing purposes, might be deleted later
        if self.dependencies != {}:
            for j in range(self.num_of_samples):
                logging.info('Generating ' + str(j + 1) + '. sample... \n')
                x = generate_samples(y, self.dependencies).copy()
                estimation += self.utility_function(x)  # might be deleted later
                # sys.stderr.write("sample is: " + str(x) + '\n')
                for i in self.dependencies.keys():  # add dependencies
                    x1 = x.copy()
                    x1[i] = 1.0
                    # sys.stderr.write("x1 is: " + str(x1) + '\n')
                    x0 = x.copy()
                    x0[i] = 0.0
                    # sys.stderr.write("x0 is: " + str(x0) + '\n')
                    grad[i] += self.utility_function(x1) - self.utility_function(x0)
                    # if grad[i] != 0.0:
                    #     sys.stderr.write("grad[" + str(i) + "] is: " + str(grad[i]) + '\n')
        else:
            for j in range(self.num_of_samples):
                logging.info('Generating ' + str(j + 1) + '. sample... \n')
                x = generate_samples(y).copy()
                estimation += self.utility_function(x)  # might be deleted later
                # sys.stderr.write("sample is: " + str(x) + '\n')
                for i in y.keys():
                    x1 = x.copy()
                    x1[i] = 1.0
                    # sys.stderr.write("x1 is: " + str(x1) + '\n')
                    x0 = x.copy()
                    x0[i] = 0.0
                    # sys.stderr.write("x0 is: " + str(x0) + '\n')
                    grad[i] += self.utility_function(x1) - self.utility_function(x0)
                    # if grad[i] != 0.0:
                    #     sys.stderr.write("grad[" + str(i) + "] is: " + str(grad[i]) + '\n')
        grad = {key: grad[key] / self.num_of_samples for key in grad.keys()}
        estimation = estimation / self.num_of_samples  # might be deleted later
        return grad, estimation


class PolynomialEstimator(GradientEstimator):
    """
    Calculates  the expectation by linearizing the function with Taylor
    approximation.
    """

    def __init__(self, my_wdnf):
        """
        my_wdnf is the resulting wdnf object after compose.
        """
        logging.info('Creating the PolynomialEstimator object...')
        super(PolynomialEstimator, self).__init__()
        self.my_wdnf = my_wdnf
        logging.info('...done.')

    def estimate(self, y):
        """
        Estimates the expectation by evaluating the resulting wdnf.
        """
        grad = dict.fromkeys(y.iterkeys(), 0.0)
        dependencies = self.my_wdnf.find_dependencies()
        for key in dependencies.keys():
            y1 = y.copy()
            y1[key] = 1
            grad1 = self.my_wdnf(y1)

            y0 = y.copy()
            y0[key] = 0
            grad0 = self.my_wdnf(y0)
            delta = grad1 - grad0
            grad[key] = delta
        estimation = self.my_wdnf(y)  # this variable will be used for testing purposes, might be deleted later
        return grad, estimation


class LinearSolver(object):  # For Python 3, replace object with ABCMeta
    """
    Abstract class to parent solver classes with different constraints.
    """
    __metaclass__ = ABCMeta  # Comment this line for Python 3

    @abstractmethod
    def __init__(self, sets, constraints):
        pass

    def solve(self, gradient):
        """
        Abstract method to solve submodular maximization problems
        according to given constraints where gradient is a GradientEstimator
        object.
        """
        pass


class UniformMatroidSolver(LinearSolver):
    """
    Given a set of elements and a cardinality constraint, creates a LinearSolver
    object. Has a solve method to-be-called in the ContinuousGreedy algorithm.
    """

    def __init__(self, ground_set, k):
        """
        groundSet is a set of elements and k is an integer denoting the
        cardinality.
        """
        logging.info('Creating UniformMatroidSolver object...')
        super(LinearSolver, self).__init__()
        self.ground_set = ground_set
        self.k = k
        logging.info('...done.')

    def solve(self, gradient):
        """
        Given a gradient dictionary with {element : partial derivative} pairs,
        returns a set of k elements from the groundSet with the maximum partial
        derivatives.
        """
        return set(nlargest(self.k, gradient, key=gradient.get))


class PartitionMatroidSolver(LinearSolver):
    """
    Given disjoint partitions of a ground set and cardinality constraints,
    creates a LinearSolver object. Has a solve method to-be-called in the
    ContinuousGreedy algorithm.
    """

    def __init__(self, partitioned_set, k_list):
        """
        partitionedSet is a dictionary of sets with {partitionName : set} pairs,
        k_list is a dictionary of cardinalities with {partitionName : int} pairs.
        """
        logging.info('Creating PartitionMatroidSolver object...')
        super(LinearSolver, self).__init__()
        self.partitioned_set = partitioned_set
        self.k_list = k_list
        logging.info('...done.')

    def solve(self, gradient):
        """
        Given a gradient dictionary with {element : partial derivative} pairs,
        returns a dictionary with {partitionName : set} pairs where each set has
        the maximum k elements from that partition in the partitionedSet with the
        maximum partial derivatives.
        """
        result = {}
        for partition in self.partitioned_set:
            uniform_solver = UniformMatroidSolver(self.partitioned_set[partition], self.k_list[partition])
            filtered_gradient = {key: gradient[key] for key in self.partitioned_set[partition]}
            selection = uniform_solver.solve(filtered_gradient)
            result[partition] = selection
        return result


class ContinuousGreedy:
    """
    Given LinearSolver and GradientEstimator objects with a initialPoint
    dictionary (fractional vector), creates a ContinuousGreedy object.
    """

    def __init__(self, linear_solver, estimator, initial_point):
        """
        linear_solver is a LinearSolver object, estimator is a GradientEstimator
        object and the initialPoint is a dictionary with {element: value} pairs
        where value is in [0, 1].
        """
        self.linear_solver = linear_solver
        self.estimator = estimator
        self.initial_point = initial_point

    def fw(self, iterations, keep_track=False):
        """
        iterations is an integer denoting the number of iterations in the
        Frank-Wolfe algorithm.
        """
        x0 = self.initial_point.copy()
        gamma = 1.0 / iterations
        y = x0.copy()
        start = time()
        track = dict()
        bases = []
        logging.info('Starting Frank-Wolfe...')
        for t in range(iterations):
            logging.info('iteration #' + str(t) + "\n")
            gradient = self.estimator.estimate(y)[0]
            mk = self.linear_solver.solve(gradient)  # finds maximum
            # sys.stderr.write("mk: " + str(mk))
            try:
                for value in mk.values():  # updates y
                    for i in value:
                        y[i] += gamma
                # y = {i: y[i] + gamma for value in mk.values() for i in value}
            except AttributeError:
                for i in mk:
                    y[i] += gamma
                # y = {i: y[i] + gamma for i in mk}
            if keep_track or t == iterations - 1:
                time_passed = time() - start
                new_y = y.copy()
                track[t] = (time_passed, new_y)
            bases.append(mk)
        # sys.stderr.write("y: " + str(y))
        return y, track, bases
