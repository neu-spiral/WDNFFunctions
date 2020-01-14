# # demands is list of object instances, edges is dictionary with (u,v) as key and mu_uv as value
#import cvxopt
import numpy as np
#import itertools
from abc import ABCMeta, abstractmethod #ABCMeta works with Python 2, use ABC for Python 3
from heapq import nlargest
from wdnf import wdnf, poly, taylor
# from time import time
# from helpers import write
# import argparse
# import os


def generateSamples(y, dependencies):
    """ Generates random samples x for e in dependencies P(x_e = 1) = y_e
    """
    samples = [0.0] * len(y)
    p = np.random.rand(len(y))
    for element in dependencies:
        if y[element] > p[element]:
            samples[element] = 1
    return samples


class GradientEstimator(object): #For Python 3, replace object with ABCMeta
    """Abstract class to parent classes of different gradient estimators.
    """
    __metaclass__ = ABCMeta #Comment out this line for Python 3


    @abstractmethod
    def __init__(self, my_wdnf, func):
        pass


    def estimate(self, y): #Should the estimate's take y as an input?
        pass


class SamplerEstimator(GradientEstimator): #
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


    def __init__(self, my_wdnf, func):
        """my_wdnf is a wdnf object and func is a function of that wdnf object
        such as func(my_wdnf) = log(my_wdnf) or
        func(my_wdnf) = my_wdnf/(1 - my_wdnf)
        """
        self.my_wdnf = my_wdnf
        self.func = func


    def estimate(self, y):
        grad = self.func(y)
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
            #for item in selection:
            #    (self.partitionedSet[partition]).pop(item, None)
            UniformSolver = UniformMatroidSolver(self.partitionedSet[partition], self.k_list[partition])
            filtered_gradient = {key: gradient[key] for key in self.partitionedSet[partition]}
            selection = UniformSolver.solve(filtered_gradient)
            result[partition] = selection
        return result


class ContinuousGreedy():
    """
    """


    def __init__(matroidSolver, estimator):
        """
        """
        self.matroidSolver = matroidSolver
        self.estimator = estimator


    def find_max(self, gradient):
        mk = (self.matroidSolver).solve(gradient)
        return mk


    def adapt(y, mk, gamma):
        for i in mk:
            y[i] += gamma


    def FW(self, iterations):
        x0 = [0.0] * len(self.y) ##How to keep the size information?
        gamma = 1.0 / iterations
        y = x0
        for t in range(iterations):
            gradient = self.estimator.estimate(y)
            mk = self.find_max(gradient)
            adapt(y, mk, gamma)
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

# class ContinuousGreedy:
#     def __init__(self, P):
#         """Construct the class, given topology (i.e., graph, demands, demands rate, bandwidths) and estimator type."""
#         self.demands = P.demands
#         dem_items=[]
#         for demand in self.demands:
#              item = demand.item
#              dem_items.append(item)
#         self.EDGE = P.EDGE # service rate for each edge
#         self.capacities = P.capacities # c_v for each node
#         self.problem_size = (len(P.capacities),len(set(dem_items)));
#         self.utilityfunction = P.utilityfunction # cost 'function'
#         self.capacity_servicerate = P.capacity_servicerate # mu_e for each edge
#         self.min_servicerate = P.min_servicerate # epsilon
#         self.k = 1
#         self.cost = P.cost
#
#     def OBJ_dict(self,ro_dict):
#         """Given rhos as a dictionary, compute the objctive."""
#         obj = 0.
#
#         for edge in ro_dict:
#             for demand in ro_dict[edge]:
#                 ro = ro_dict[edge][demand]
#                 obj = obj+self.utilityfunction(ro,0)
#         return obj
#
#     def Dependencies(self):
#         """Find which edges an element (v,i) affects, in general."""
#         dep = {}
#
#         for demand in self.demands:
#             path = demand.path
#             item = demand.item
#             path_len = len(path)
#             if path_len > 1:
#                 v = path[path_len-2]
#                 depend = [(path[path_len-2],path[path_len-1])]
#                 if (v,item) in dep:
#                     dep[(v,item)][demand] = depend
#                 else:
#                     dep[(v,item)] = {demand: depend}
#                 for i in range(path_len-3,-1,-1):
#                     v = path[i]
#                     u = path[i + 1]
#                     depend = [(v, u)] + dep[(u, item)][demand]
#                     if (v,item) in dep:
#                         dep[(v,item)][demand]= depend
#                     else:
#                         dep[(v,item)] = {demand: depend}
#             else:
#                 pass
#         return dep
#
#
#  # search for m?
#     def find_max(self,Z):
#         """Given the gradient as a matrix Z, find the argument of its top K values in each row. K is the capacity of each row."""
#         def topK(row, K):
#             top_K = []
#             for i in range(K):
#                 iStar = np.argmax(row)
#                 top_K.append(iStar)
#                 row[iStar]= -1. #Make sure it is not going to be picked again!
#             return top_K
#         V,I = self.problem_size
#         D = {}
#         for v in range(V):
#             D[v] = topK(Z[v,:],self.capacities[v])
#         return D
#
#     def find_max_Mu(self, Gradient_Mu):
#         '''Given the gradient, find the request corresponding to the greatest gradient for each edge'''
#         Direction_Mu = {}
#         for edge in self.capacity_servicerate:
#             Direction_Mu[edge] = None
#         for edge in self.capacity_servicerate:
#             if Gradient_Mu.has_key(edge):
#                 list_demands = Gradient_Mu[edge].keys()
#                 list_gradient = Gradient_Mu[edge].values()
#                 index = list_gradient.index(max(list_gradient))
#                 Direction_Mu[edge] = {}
#                 Direction_Mu[edge]['index']=list_demands[index]
#                 Direction_Mu[edge]['value']=self.capacity_servicerate[edge] - self.min_servicerate*len(list_demands)
#             else:
#                 pass
#         return Direction_Mu
#
#     def FW(self, ITERATIONS, debug_mod):
#         """Implement the continiuous greedy algorithm."""
#         def adapt(Y,D,gamma):
#             for v in D:
#                 for i in D[v]:
#                     Y[v,i]+=gamma
#
#         def adapt_Mu(Direction_Mu, gamma):
#             for edge in Direction_Mu:
#                 if Direction_Mu[edge] != None:
#                     demand = Direction_Mu[edge]['index']
#                     self.EDGE[edge][demand] += gamma*Direction_Mu[edge]['value']
#                 else:
#                     pass
#
#         V,I = self.problem_size # number of nodes, number of items
#         Y = cvxopt.matrix(0.,(V,I))
#         gamma = 1./ITERATIONS
#
#         ro_dict = self.ro_uvr_CONT(Y)
#         self.cost = self.OBJ_dict(ro_dict)
#         obj = self.cost
#         dependencies =  self.Dependencies()
#         start = time()
#         track = [self.cost]
#         track += [(start-start,self.cost-obj)]
#         bases = []
#         for t in range(ITERATIONS):
#             [Z, Gradient_Mu] = self.Estimate_Gradient(Y,dependencies) # a gradient estimate for X and mu separately
#             D = self.find_max(Z) # a node*capacity dictionary, direction of update
#             Direction_Mu = self.find_max_Mu(Gradient_Mu)
#
#             adapt(Y,D,gamma)
#             adapt_Mu(Direction_Mu,gamma)
#             if debug_mod or t==ITERATIONS-1:
#
#                 ro_dict = self.ro_uvr_CONT(Y)
#                 cost = self.OBJ_dict(ro_dict)
#                 obj = self.cost - cost
#                 current = time() - start
#                 track.append((current,obj))
#                 print "Objective is: ",obj," time taken: ",current
#             bases.append(D)
#
#         return Y, track, bases
#
#     def ro_uvr_CONT(self, Y):
#         """Compute  rho for each edge and demand, given a fractional Y."""
#         ro_uvr = {}
#         # Go through demands
#         for demand in self.demands:
#             path = demand.path
#             item = demand.item
#             rate = demand.rate
#
#             prod_i = 1.
#             for node_i in range(1,len(path)):
#                 y = Y[path[node_i-1],item]
#                 if y < 1.0:
#                     prod_i *= (1.-y)
#                     edge = (path[node_i-1],path[node_i])
#                     ro = rate*prod_i/self.EDGE[edge][demand]
#                 else:
#                     break
#                 if edge in ro_uvr:
#                     ro_uvr[edge][demand] = ro
#                 else:
#                     ro_uvr[edge]={demand: ro}
#         return ro_uvr
#
#     def RhoPoly(self):
#         rho_uvs_coefficients,  rho_uvs_sets = rho_uv_dicts(self.demands, self.EDGE)
#         rho_poly = {}
#         for edge in rho_uvs_coefficients:
#             rho_poly[edge]  = {}
#             for demand in rho_uvs_coefficients[edge]:
#                 rho_poly[edge][demand] = {}
#                 rho_poly[edge][demand][1] = poly(rho_uvs_coefficients[edge][demand], rho_uvs_sets[edge][demand]) # construct rho_e class
#                 for i in range(2, self.k+1):
#                     rho_poly[edge][demand][i] = rho_poly[edge][demand][1].power(i)
#         return rho_poly
#
#
#
#class ContinuousGreedy_sampling(ContinuousGreedy):
#     def __init__(self, P, num_samples):
#         ContinuousGreedy.__init__(self,P=P)
#         self.num_samples = num_samples
#
#     def Estimate_Gradient(self, Y, dependencies):
#         """Compute the gradient by sampling."""
#
#         def gen_samp(Y,dependencies):
#             V,I = Y.size
#             X_sampled = cvxopt.matrix(0,(V,I))
#             P = np.random.rand(V,I)
#             for edge in dependencies:
#                 (v,i) = edge
#                 if P[v,i] < Y[v,i]:
#                     X_sampled[v,i] = 1
#             return X_sampled
#
#         (V,I) = np.matrix(Y).shape
#         grad_Y = cvxopt.matrix(0.,(V,I))
#
#         grad_Mu = {}
#         for edge in self.EDGE:
#             grad_Mu[edge] = {}
#             for demand in self.EDGE[edge]:
#                 grad_Mu[edge][demand] = 0.0
#
#         rho_poly = self.RhoPoly() # Update rho for each iteration of FW
#         for k in range(self.num_samples):
#
#             X_sampled = gen_samp(Y,dependencies)
#             '''Calculate gradient w.r.t. Y'''
#             for (v,i) in dependencies:
# #                Y1 = +X_sampled
# #                Y1[v,i] = 1
#                 Y0 = +X_sampled
#                 Y0[v,i] = 0
#
#                 for demand in dependencies[(v,i)]:
#                     for edge in dependencies[(v,i)][demand]:
#                         r_polys = rho_poly[edge][demand]
#                         r0 = r_polys[1].evaluate(Y0)
# #                        r1 = r_polys[1].evaluate(X_sampled)
# #                        if r0 != r1:
# #                            print r1
# #                        r1 = r_polys[1].evaluate(Y1)
# #                        delta = self.utilityfunction(r0, 0) - self.utilityfunction(r1, 0)
#                         delta = self.utilityfunction(r0, 0)
#                         grad_Y[v, i] += delta
#             '''
#             for (v,i) in dependencies:
#                 Y1 = +X_sampled
#                 Y1[v,i] = 1
#                 Y0 = +X_sampled
#                 Y0[v, i] = 0
#                 ro_dict1 = self.ro_uvr_CONT(Y1)
#                 obj1 = self.OBJ_dict(ro_dict1)
#                 ro_dict0 = self.ro_uvr_CONT(Y0)
#                 obj0 = self.OBJ_dict(ro_dict0)
#                 grad_Y[v,i] += obj0 - obj1
#             '''
#
#
#             '''Calculate gradient w.r.t. Mu'''
#             for demand in self.demands:
#                 path = demand.path
#                 path_len = len(path)
#                 for node in range(path_len - 1):
#                     edge = (path[node], path[node + 1])
#                     r_polys = rho_poly[edge][demand]
#                     r = r_polys[1].evaluate(X_sampled)
#                     delta = self.utilityfunction(r, 1) * r / self.EDGE[edge][demand]
#                     grad_Mu[edge][demand] += delta
#
#         '''Average gradient w.r.t. Y'''
#         grad_Y = grad_Y / self.num_samples
#         '''Average gradient w.r.t. Mu'''
#         for edge in grad_Mu:
#             for demand in grad_Mu[edge]:
#                 grad_Mu[edge][demand] = grad_Mu[edge][demand]/self.num_samples
#
#         return grad_Y, grad_Mu
#
#
# class ContGreedyPoly(ContinuousGreedy):
#     """A class executing continuous greedy using the Taylor expansion of E_Y[rho(X)] of order k
#     """
#
#     def __init__(self, P, k):
#         ContinuousGreedy.__init__(self,P)
#         self.k = k
#
#     def Estimate_Gradient(self, Y, dependencies):
#         """ Estimate the gradient using a taylor approximation of E[rho(X)]
#
#         """
#
#         '''Update rho for each iteration of FW'''
#         rho_poly = self.RhoPoly()
#
#         (V,I) = np.matrix(Y).shape
#         grad_Y = cvxopt.matrix(0.,(V,I))
#
#         '''Calculate gradient w.r.t. Y'''
#         for (v,i) in dependencies:
# #            Y1 = +Y
# #            Y1[v,i] = 1
#             Y0 = +Y
#             Y0[v,i] = 0
#             for demand in dependencies[(v,i)]:
#                 for edge in dependencies[(v,i)][demand]:
#
#                     r_polys = rho_poly[edge][demand]
#                     r = r_polys[1].evaluate(Y) # rho_e(Y) = rho^*
#                     derivative_value = dict( [(ii, self.utilityfunction(r,ii)) for ii in range(self.k+1)]) # derivative of cost
#                     approx = taylor(derivative_value, r, self.k) # taylor coefficient
# #                    r1 = r_polys.evaluate(Y1)
#
# #                    r_powers1 = dict([ (j,r1**j) for j in range(1,self.k+1) ])
#                     r_powers0 = dict([ (j,r_polys[j].evaluate(Y0)) for j in range(1,self.k+1) ])
# #                    r_powers0 = dict([ (j,(r_polys[1].evaluate(Y0))**j) for j in range(1,self.k+1) ])
#
# #                    delta = approx.evaluate_expanded(r_powers0) - approx.evaluate_expanded(r_powers1)
#                     delta = approx.evaluate_expanded(r_powers0)
#                     grad_Y[v,i] += delta
#
#         grad_Mu = {}
#         for edge in self.EDGE:
#             grad_Mu[edge] = {}
#             for demand in self.EDGE[edge]:
#                 grad_Mu[edge][demand] = 0.0
#
#         '''Calculate gradient w.r.t. Mu'''
#         for demand in self.demands:
#             path = demand.path
#             path_len = len(path)
#             for node in range(path_len-1):
#                 edge = (path[node], path[node+1])
#                 r_polys = rho_poly[edge][demand]
#                 r = r_polys[1].evaluate(Y)
#                 derivative_value = dict( [(ii, self.utilityfunction(r,ii)) for ii in range(self.k+1)])
#                 approx = taylor(derivative_value, r, self.k)
#                 r_powers = dict( [ (j,r_polys[j].evaluate(Y)) for j in range(1,self.k+1)] )
# #                r_powers = dict( [ (j,(r_polys[1].evaluate(Y))**j) for j in range(1,self.k+1)] )
#
#                 delta = approx.evaluate_expanded_mu(r_powers)/self.EDGE[edge][demand]
#                 grad_Mu[edge][demand] = delta
#         return grad_Y, grad_Mu
#
#
# class ContGreedyPower(ContinuousGreedy):
#     def __init__(self, P):
#         ContinuousGreedy.__init__(self,P)
#
#
#     def Estimate_Gradient(self,Y,dependencies):
#         """ Estimate the gradient using the power approcximation of E[c(rho(X))] = rho
#
#         """
#
#         rho_poly = self.RhoPoly()
#
#         (V,I) = self.problem_size
#         grad_Y = cvxopt.matrix(0.,(V,I))
#
#         for (v,i) in dependencies:
# #            Y1 = +Y
# #            Y1[v, i] = 1.
#             Y0 = +Y
#             Y0[v, i] = 0.
#             for demand in dependencies[(v,i)]:
#                 for edge in dependencies[(v,i)][demand]:
#                     r_polys = rho_poly[edge][demand]
#                     r0 = r_polys[1].evaluate(Y0)
# #                    r1 = r_polys.evaluate(Y1)
# #                    delta = self.utilityfunction(r0,0) - self.utilityfunction(r1,0)
#                     delta = self.utilityfunction(r0,0)
#                     grad_Y[v,i] += delta
#
#         grad_Mu = {}
#         for edge in self.EDGE:
#             grad_Mu[edge] = {}
#             for demand in self.EDGE[edge]:
#                 grad_Mu[edge][demand] = 0.0
#         for demand in self.demands:
#             path = demand.path
#             path_len = len(path)
#             for node in range(path_len-1):
#                 edge = (path[node], path[node+1])
#                 r_polys = rho_poly[edge][demand]
#                 r = r_polys[1].evaluate(Y)
#                 delta = self.utilityfunction(r,1) * r / self.EDGE[edge][demand]
#                 grad_Mu[edge][demand] = delta
#         return grad_Y, grad_Mu
#
#
# class DRSubmodular(ContinuousGreedy):
#     def __init__(self, P):
#         ContinuousGreedy.__init__(self,P)
#
#
#     def Estimate_Gradient(self,Y,dependencies):
#         """ Calculate the gradient by chain rule
#
#         """
#
#         rho_poly = self.RhoPoly()
#
#         (V,I) = self.problem_size
#         grad_Y = cvxopt.matrix(0.,(V,I))
#
#         '''Calculate gradient w.r.t. '''
#         for (v,i) in dependencies:
# #            Y1 = +Y
# #            Y1[v, i] = 1.
#             Y0 = +Y
#             Y0[v, i] = 0.
#             for demand in dependencies[(v,i)]:
#                 for edge in dependencies[(v,i)][demand]:
#                     r_polys = rho_poly[edge][demand]
#                     r = r_polys[1].evaluate(Y)
#                     r0 = r_polys[1].evaluate(Y0)
# #                    r1 = r_polys.evaluate(Y1)
# #                    delta = self.utilityfunction(r0,0) - self.utilityfunction(r1,0)
#                     delta = self.utilityfunction(r,1) * r0
#                     grad_Y[v,i] += delta
#
#         grad_Mu = {}
#         for edge in self.EDGE:
#             grad_Mu[edge] = {}
#             for demand in self.EDGE[edge]:
#                 grad_Mu[edge][demand] = 0.0
#
#         '''Calculate gradient w.r.t. Mu'''
#         for demand in self.demands:
#             path = demand.path
#             path_len = len(path)
#             for node in range(path_len-1):
#                 edge = (path[node], path[node+1])
#                 r_polys = rho_poly[edge][demand]
#                 r = r_polys[1].evaluate(Y)
#                 delta = self.utilityfunction(r,1) * r / self.EDGE[edge][demand]
#                 grad_Mu[edge][demand] = delta
#         return grad_Y, grad_Mu
