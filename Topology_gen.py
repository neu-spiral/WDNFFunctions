#! /usr/bin/env python
'''
	A Cache Network
'''
from abc import ABCMeta, abstractmethod
# from Caches import PriorityCache, EWMACache, LMinimalCache
from networkx import Graph, DiGraph, shortest_path
import networkx
import random
from cvxopt import spmatrix, matrix
from cvxopt.solvers import lp
from simpy import *
from scipy.stats import rv_discrete
import numpy as np
from numpy.linalg import matrix_rank
import logging, argparse
import itertools
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import topologies
import os
from UtilityFunction import *


class CONFIG(object):
    QUERY_MESSAGE_LENGTH = 0.0
    RESPONSE_MESSAGE_LENGTH = 0.0
    EXPLORE_MESSAGE_LENGTH = 0.0
    EXPLORE_RESPONSE_MESSAGE_LENGTH = 0.0


def pp(l):
    return ' '.join(map(str, l))


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


class Problem:
    def __init__(self, capacities, demands, bandwidths, utilityfunction, capacity_servicerate, min_servicerate, cost=0):
        self.capacities = capacities
        self.EDGE = bandwidths
        self.demands = demands
        self.capacity_servicerate = capacity_servicerate
        self.min_servicerate = min_servicerate
        self.utilityfunction = utilityfunction
        self.cost = cost

    def pickle_cls(self, fname):
        f = open(fname, 'w')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def unpickle_cls(fname):
        with file(fname, 'r') as f:
            return pickle.load(f)


def generate_bandwidths(demands, min_servicerate):
    # A list to generate service rate for each edge having demand going through
    EDGE_mu = {}
    for demand in demands:
        path = demand.path
        for node in range(len(path)-1):
            edge = (path[node], path[node+1])
            '''Construct a dictionary if this EDFE_mu[edge] does not exist'''
            if edge in EDGE_mu:
                EDGE_mu[edge][demand] = min_servicerate
            else:
                EDGE_mu[edge] = {demand: min_servicerate}

    return EDGE_mu


def main():
    # logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Simulate a Network of Caches',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outputfile', help='Output file')
    parser.add_argument('--queue_type', default='MMInfty', type=str, help='Type of queue', choices=['MMInfty', 'Info'])
    parser.add_argument('--order_moment', default='2', type=str, help='Order of moment for the expected cost', choices=['1','2','3','4'])
    parser.add_argument('--min_servicerate', default=0.1, type=float, help='Minimum service rate per edge')
    parser.add_argument('--min_capacity_servicerate', default=200.0, type=float, help='Minimum capacity of service rate per edge')
    parser.add_argument('--max_capacity_servicerate', default=200.0, type=float, help="Maximum capacity of service rate per edge")
    parser.add_argument('--max_capacity', default=2, type=int, help='Maximum capacity per cache')
    parser.add_argument('--min_capacity', default=2, type=int, help='Minimum capacity per cache')

    parser.add_argument('--max_rate', default=2.0, type=float, help='Maximum demand rate')
    parser.add_argument('--min_rate', default=1.0, type=float, help='Minimum demand rate')
    parser.add_argument('--catalog_size', default=100, type=int, help='Catalog size')
    parser.add_argument('--demand_size', default=1000, type=int, help='Demand size')
    parser.add_argument('--demand_distribution', default="powerlaw", type=str, help='Demand distribution',
                        choices=['powerlaw', 'uniform'])
    parser.add_argument('--powerlaw_exp', default=1.2, type=float,
                        help='Power law exponent, used in demand distribution')
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--graph_size', default=100, type=int, help='Network size')
    parser.add_argument('--graph_degree', default=4, type=int,
                        help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p', default=0.10, type=int, help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('--random_seed', default=4156910908, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)

    def graphGenerator():
        if args.graph_type == "erdos_renyi":
            return networkx.erdos_renyi_graph(args.graph_size, args.graph_p)
        if args.graph_type == "balanced_tree":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(args.graph_degree)))
            return networkx.balanced_tree(args.graph_degree, ndim)
        if args.graph_type == "cicular_ladder":
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.circular_ladder_graph(ndim)
        if args.graph_type == "cycle":
            return networkx.cycle_graph(args.graph_size)
        if args.graph_type == 'grid_2d':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim, ndim)
        if args.graph_type == 'lollipop':
            ndim = int(np.ceil(args.graph_size * 0.5))
            return networkx.lollipop_graph(ndim, ndim)
        if args.graph_type == 'expander':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.margulis_gabber_galil_graph(ndim)
        if args.graph_type == "hypercube":
            ndim = int(np.ceil(np.log(args.graph_size) / np.log(2.0)))
            return networkx.hypercube_graph(ndim)
        if args.graph_type == "star":
            ndim = args.graph_size - 1
            return networkx.star_graph(ndim)
        if args.graph_type == 'barabasi_albert':
            return networkx.barabasi_albert_graph(args.graph_size, args.graph_degree)
        if args.graph_type == 'watts_strogatz':
            return networkx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
        if args.graph_type == 'regular':
            return networkx.random_regular_graph(args.graph_degree, args.graph_size)
        if args.graph_type == 'powerlaw_tree':
            return networkx.random_powerlaw_tree(args.graph_size)
        if args.graph_type == 'small_world':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.navigable_small_world_graph(ndim)
        if args.graph_type == 'geant':
            return topologies.GEANT()
        if args.graph_type == 'dtelekom':
            return topologies.Dtelekom()
        if args.graph_type == 'abilene':
            return topologies.Abilene()
        if args.graph_type == 'servicenetwork':
            return topologies.ServiceNetwork()

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2015)

    construct_stats = {}  # to store properties of network

    logging.info('Generating graph and weights...')
    temp_graph = graphGenerator()  # use networkx to generate a graph
    # networkx.draw(temp_graph)
    # plt.draw()
    V = len(temp_graph.nodes())
    E = len(temp_graph.edges())
    print V, E
    logging.debug('nodes: ' + str(temp_graph.nodes()))  # list
    logging.debug('edges: ' + str(temp_graph.edges()))  # list of node pair
    G = DiGraph()  # generate a DiGraph

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())  # add node from temp_graph to G
    for (x, y) in temp_graph.edges():  # add edge from temp_graph to G
        xx = number_map[x]
        yy = number_map[y]
        G.add_edges_from(((xx, yy), (yy, xx)))
    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size, edge_size))
    logging.debug('G is:' + str(G.nodes()) + str(G.edges()))
    construct_stats['graph_size'] = graph_size
    construct_stats['edge_size'] = edge_size
    logging.info('Generating item sources...')
    item_sources = dict((item, [G.nodes()[source]]) for item, source in zip(range(args.catalog_size),
                                                                            np.random.choice(range(graph_size),
                                                                                             args.catalog_size)))  # generate designated servers for each object catalog
    logging.info('...done. Generated %d sources' % len(item_sources))
    logging.debug('Generated sources:')
    for item in item_sources:
        logging.debug(pp([item, ':', item_sources[item]]))

    construct_stats['sources'] = len(item_sources)

    logging.info('Generating query node list...')
    query_node_list = [G.nodes()[i] for i in random.sample(xrange(graph_size),
                                                           args.query_nodes)]  # generate a list of nodes which query requests
    logging.info('...done. Generated %d query nodes.' % len(query_node_list))

    construct_stats['query_nodes'] = len(query_node_list)

    logging.info('Generating demands...')
    if args.demand_distribution == 'powerlaw':
        factor = lambda i: (1.0 + i) ** (-args.powerlaw_exp)
    else:
        factor = lambda i: 1.0
    pmf = np.array([factor(i) for i in range(args.catalog_size)])
    pmf /= sum(pmf)  # probability of quering each item
    distr = rv_discrete(values=(range(args.catalog_size), pmf))
    if args.catalog_size < args.demand_size:
        items_requested = list(distr.rvs(size=(args.demand_size - args.catalog_size))) + range(
            args.catalog_size)  # Why?
    else:
        items_requested = list(distr.rvs(size=args.demand_size))

    random.shuffle(items_requested)  # a list of item which queried by each demand

    demands_per_query_node = args.demand_size // args.query_nodes
    remainder = args.demand_size % args.query_nodes
    demands = []  # list of Demand(item queried, path, rate)
    for x in query_node_list:
        dem = demands_per_query_node
        if x < remainder:
            dem = dem + 1  # number of demands x queries

        new_dems = [
            Demand(items_requested[pos], shortest_path(G, x, item_sources[items_requested[pos]][0], weight='weight'),
                   random.uniform(args.min_rate, args.max_rate)) for pos in range(len(demands), len(demands) + dem)]
        logging.debug(pp(new_dems))
        demands = demands + new_dems

    #Demand_list = []
    #for demand in demands:
    #    Demand_list.append(demand.__dict__)

    # demands = Demand_list  # Why we need class's attributes other than class
    logging.info('...done. Generated %d demands' % len(demands))
    # plt.hist([ d.item for d in demands], bins=np.arange(args.catalog_size)+0.5)
    # plt.show()
    construct_stats['demands'] = len(demands)

    logging.info('Generating capacities...')
    capacities = dict(
        (x, random.randint(args.min_capacity, args.max_capacity)) for x in G.nodes())  # Cache capacity for each node
    logging.info('...done. Generated %d caches' % len(capacities))
    logging.debug('Generated capacities:')
    for key in capacities:
        logging.debug(pp([key, ':', capacities[key]]))


    logging.info('Generating capacities of service rate...')
    capacity_servicerate = dict( (x, random.uniform(args.min_capacity_servicerate, args.max_capacity_servicerate) ) for x in G.edges())
    logging.info('...done. Generated %d service rate capacities' % len(capacity_servicerate))
    logging.debug('Generated capacities of service rate:')
    for key in capacity_servicerate:
        logging.debug(pp([capacity_servicerate, ':', capacity_servicerate[key]]))

    bandwidths = generate_bandwidths(demands, args.min_servicerate)


    logging.info('Building CacheNetwork')
    logging.info('...done')

    dir = "INPUT/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    out = dir + args.outputfile + "_" + args.graph_type + "_" + str(args.demand_size) + "demands_" + str(args.catalog_size) + "catalog_size_" + str(args.min_capacity) + "mincap_"  + str(args.max_capacity) + "maxcap_"  + str(args.graph_size) + "size_" + str(args.demand_distribution) + "_" + "rate" + str(args.min_rate) + "_" + str(args.query_nodes) + "qnodes_" + args.order_moment + "order_" + args.queue_type

    functionchoice = args.order_moment+args.queue_type
    if functionchoice == '1MMInfty':
        utilityfunction = UtilityFunction1
    elif functionchoice == '2MMInfty':
        utilityfunction = UtilityFunction2
    elif functionchoice == '3MMInfty':
        utilityfunction = UtilityFunction3
    elif functionchoice == '4MMInfty':
        utilityfunction = UtilityFunction4
    elif functionchoice == '1Info':
        utilityfunction = UtilityFunction5
    elif functionchoice == '2Info':
        utilityfunction = UtilityFunction6
    elif functionchoice == '3Info':
        utilityfunction = UtilityFunction7
    elif functionchoice == '4Info':
        utilityfunction = UtilityFunction8
    else:
        utilityfunction = None


    pr = Problem(capacities, demands, bandwidths, utilityfunction, capacity_servicerate, args.min_servicerate)  # pack the graph, capacity for each node, attributes of each demands(requests), service rate for each edge

    pr.pickle_cls(out) # can only pickle functions defined at the top level of a module


if __name__ == "__main__":
    main()
