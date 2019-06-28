#! /usr/bin/env python
'''
	A Cache Network
'''
from abc import ABCMeta, abstractmethod
from Caches import PriorityCache, EWMACache, LMinimalCache
from networkx import Graph, DiGraph, shortest_path
from cvxopt import spmatrix, matrix
from cvxopt.solvers import lp
from simpy import *
import numpy as np
from numpy.linalg import matrix_rank
import logging, argparse
import itertools
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
from Toplogy_gen import Problem, Demand
from queuing import *
import os
import matplotlib.pyplot as plt




class CONFIG(object):
    QUERY_MESSAGE_LENGTH = 0.0
    RESPONSE_MESSAGE_LENGTH = 0.0
    EXPLORE_MESSAGE_LENGTH = 0.0
    EXPLORE_RESPONSE_MESSAGE_LENGTH = 0.0


def pp(l):
    return ' '.join(map(str, l))


class Message(object):
    """A Message object.

       Attributes:
           header: the header of the message (e.g., query_message, response_message
           payload: the payload, can be set by the programmer
           length: length, to be used in transmission delay calculations
           stats: statistics collected as the message traverses nodes
       u_bit: indicates if the message is going upstream or downstream
    """

    def __init__(self, header, payload, length, stats, u_bit, counter=1):
        self.header = header
        self.payload = payload
        self.length = length
        self.u_bit = u_bit
        self.counter = counter # record how many requests are merged into one
        if stats == None:
            self.stats = {}
            self.stats['spawned_time'] = None
            self.stats['delay'] = 0.0
            self.stats['hops'] = 0.0
        else:
            self.stats = stats

    def __str__(self):
        return Message.__repr__(self)

    def __repr__(self):
        return pp(['Message(', self.header, ',', self.payload, ',', self.length, ',', self.stats, ')'])


class QueryMessage(Message):
    """
     A query message.
    """

    def __init__(self, d, query_id, stats=None):
        Message.__init__(self, header=("QUERY", d, query_id), payload=[["QUERY", d, query_id]], length=CONFIG.QUERY_MESSAGE_LENGTH,
                         stats=stats, u_bit=True)


class ResponseMessage(Message):
    """
     A response message.
    """

    def __init__(self, d, query_id, payload, stats=None):
        Message.__init__(self, header=("RESPONSE", d, query_id), payload=payload, length=CONFIG.RESPONSE_MESSAGE_LENGTH,
                         stats=stats, u_bit=False)



class CacheNetwork(DiGraph):
    """A cache network.

      A cache network comprises a weighted graph and a list of demands. Each node in the graph is associated with a cache of finite capacity.
      NetworkCaches must support a message receive operation, that determines how they handle incoming messages.

      The cache networks handles messaging using simpy stores and processes. In partiqular, each cache, edge and demand is associated with a
      Store object, that receives, stores, and processes messages from simpy processes.

      In more detail:
      - Each demand is associated with two processes, one that generates new queries, and one that monitors and logs completed queries (existing only for logging purposes)
      - Each cache/node is associated with a process that receives messages, and processes them, and produces new messages to be routed, e.g., towards neigboring edges
      - Each edge is associated with a process that receives messages to be routed over the edge, and delivers them to the appropriate target node.
        During "delivery", messages are (a) delayed, according to configuration parameters, and (b) statistics about them are logged (e.g., number of hops, etc.)

      Finally, a global monitoring process computes the social welfare at poisson time intervals.

    """

    def __init__(self, G, cacheGenerator, demands, item_sources, capacities, weights, delays, utilityfunction_n, utilityfunction_rho, X, Queuing_type, warmup=0,
                 monitoring_rate=1.0):
        self.env = Environment()
        self.warmup = warmup
        self.demandstats = {}
        self.sw = {}
        self.funstats = {}
        self.monitoring_rate = monitoring_rate
        self.delays = delays
        self.utilityfunction_n = utilityfunction_n
        self.utilityfunction_rho = utilityfunction_rho

        DiGraph.__init__(self, G)
        for x in self.nodes():
            self.node[x]['cache'] = cacheGenerator(capacities[x], x, X[x,:])
            self.node[x]['pipe'] = Store(self.env)
        for e in self.edges():
            x = e[0]
            y = e[1]
            if e in delays:
                self.edge[x][y]['delay'] = delays[e].copy()
                self.edge[x][y]['pipe_query'] = SimpleQueue(self.env, self.edge[x][y]['delay'], self.node[y]['pipe'])
                self.edge[y][x]['delay'] = delays[e].copy()
                self.edge[y][x]['pipe_response'] = MultiQueue(self.env, Queuing_type, self.edge[x][y]['delay'], self.node[y]['pipe'])

        self.demands = {}
        self.item_set = set()

        for d in demands:
            self.demands[d] = {}
            self.demands[d]['pipe'] = Store(self.env)
            self.demands[d]['queries_spawned'] = 0L
            self.demands[d]['queries_satisfied'] = 0L
            self.demands[d]['queries_logged'] = 0.0
            self.demands[d]['pending'] = set([])
            self.demands[d]['stats'] = {}
            self.item_set.add(d.item)

        for item in item_sources:
            for source in item_sources[item]:
                self.node[source]['cache'].makePermanent(item)  ###THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE

        for d in self.demands:
            self.env.process(self.spawn_queries_process(d))
            self.env.process(self.demand_monitor_process(d))

        for x in self.nodes():
            self.env.process(self.cache_process(x))

        self.env.process(self.monitor_process())

    def run(self, finish_time):

        logging.info('Simulating..')
        self.env.run(until=finish_time)
        logging.info('..done simulating')

    def spawn_queries_process(self, d):
        """ A process that spawns queries.

            Queries are generated according to a Poisson process with the appropriate rate. Queries generated are pushed to the query source node.
            """
        while True:
            logging.debug(pp([self.env.now, ':New query for', d.item, 'to follow', d.path]))
            _id = self.demands[d]['queries_spawned']
            qm = QueryMessage(d, _id)  # create a new query message at the query_source
            qm.stats['spawned_time'] = self.env.now  # record the born time of generated message
            self.demands[d]['pending'].add(_id) # record ids for generated queries 
            self.demands[d]['queries_spawned'] += 1
            yield self.node[d.query_source]['pipe'].put((qm, (d, d.query_source))) # is it necessary to use yield?
            yield self.env.timeout(random.expovariate(d.rate))

    def demand_monitor_process(self, d):
        """ A process monitoring statistics about completed requests.
        """
        while True:
            msg = yield self.demands[d]['pipe'].get()

            lab, dem, query_id = msg.header
            stats = msg.stats
            now = self.env.now
            # check whether it is response
            if lab is not "RESPONSE":
                logging.warning(pp([now, ': ', d, 'received a non-response message:', msg]))
                continue
            # check whether it is demand queried by this demand
            if dem is not d:
                logging.warning(pp([now, ': ', d, 'received a message', msg, 'aimed for demand', dem]))
                continue
            # acknowledge that this request gets the response
            if query_id not in self.demands[d]['pending']:
                logging.warning(pp(['Query', query_id, 'of', d, 'satisfied but not pending']))
                continue
            else:
                for i in range(msg.counter):
                    id = msg.payload[i][2] # query_id
                    self.demands[d]['pending'].remove(id)
                    logging.debug(pp(['Query', id, 'of', d, 'satisfied with stats', stats]))
                    self.demands[d]['queries_satisfied'] += 1L

            if now >= self.warmup:
                self.demands[d]['queries_logged'] += msg.counter
                #  msg.stats:'delay''hops'
                for key in stats:
                    if key in self.demands[d]['stats']:
                        self.demands[d]['stats'][key] += stats[key]
                    else:
                        self.demands[d]['stats'][key] = stats[key]


    def cache_process(self, x):
        """A process handling messages sent to caches.

           It is effectively a wrapper for a receive call, made to a NetworkedCache object.

        """
        while True:
            message = yield self.node[x]['pipe'].get() # e: (demand, demand.query_source)
            (msg, e) = message
            # update stats of msg
            msg.stats['delay'] = self.env.now - msg.stats['spawned_time']
            msg.stats['hops'] += 1
            generated_messages = self.node[e[1]]['cache'].receive(msg, e,
                                                                  self.env.now)  # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!!
            for (new_msg, new_e) in generated_messages:
                if new_e[1] in self.demands: # this message is sent to its query node
                    yield self.demands[new_e[1]]['pipe'].put(new_msg) 
                else:
                    label, d, query_id = new_msg.header
                    if label == "QUERY":
                        self.edge[new_e[0]][new_e[1]]['pipe_query'].enqueue((new_msg, new_e))
                    elif label == "RESPONSE":
                        self.edge[new_e[0]][new_e[1]]['pipe_response'].enqueue((new_msg, new_e), d)

    def cachesToMatrix(self):
        """Constructs a matrix containing cache information.
        """
        zipped = []
        n = len(self.nodes())
        m = max(len(self.item_set), max(self.item_set) + 1)
        for x in self.nodes():
            for item in self.node[x]['cache']:
                zipped.append((1, x, item))

        val, I, J = zip(*zipped)

        return spmatrix(val, I, J, size=(n, m)) # a node*item matrix whose element = 1 if x_{IJ} = 1


    def social_welfare(self):
        """ Function computing the social welfare.
        """
        utility = 0.0
        for e in self.edges():
            x = e[0]
            y = e[1]
            if self.edge[x][y].has_key('pipe_response'):
                for d in self.edge[x][y]['pipe_response'].queues:
                    queuing_size = self.edge[x][y]['pipe_response'].queues[d].get_custormers()
                    utility += self.utilityfunction_n(queuing_size)
        return utility


    def cost_without_caching(self):
        """ Function computing the  cost of recovering all items demanded from respective sources."""
 
        ro_uvr = {}
        # Go through demands
        for demand in self.demands:
            path = demand.path
            item = demand.item
            rate = demand.rate

            prod_i = 1.
            for node_i in range(1,len(path)):
                edge = (path[node_i-1],path[node_i])
                ro = rate/self.delays[edge][demand]
                if edge in ro_uvr:
                    ro_uvr[edge][demand] = ro
                else:
                    ro_uvr[edge]={demand: ro}

        obj = 0.

        for edge in ro_uvr:
            for demand in ro_uvr[edge]:
                ro = ro_uvr[edge][demand]
                obj = obj+self.utilityfunction_rho(ro,0)
        
        return obj


    def expected_caching_cost(self, Y):
        """ Function computing the expected caching gain under marginals Y, presuming product form. Also computes deterministic caching gain if Y is integral.
        """
        ro_uvr = {}
        # Go through demands
        for demand in self.demands:
            path = demand.path
            item = demand.item
            rate = demand.rate

            prod_i = 1.
            for node_i in range(1,len(path)):
                y = Y[path[node_i-1],item]
                if y < 1.0:
                    prod_i *= (1.-y)
                    edge = (path[node_i-1],path[node_i])
                    ro = rate*prod_i/self.delays[edge][demand]
                else:
                    break
                if edge in ro_uvr:
                    ro_uvr[edge][demand] = ro
                else:
                    ro_uvr[edge]={demand: ro}

        obj = 0.

        for edge in ro_uvr:
            for demand in ro_uvr[edge]:
                ro = ro_uvr[edge][demand]
                obj = obj+self.utilityfunction_rho(ro,0)
        
        return obj


    def demand_stats(self):
        """ Computed stats across demands.
        """
        stats = {}
        queries_logged = 0.0
        for d in self.demands:
            queries_logged += self.demands[d]['queries_logged']
            for key in self.demands[d]['stats']:
                if key in stats:
                    stats[key] += self.demands[d]['stats'][key]
                else:
                    stats[key] = self.demands[d]['stats'][key]
        for key in stats:
            stats[key] = stats[key] / queries_logged

        return stats

    def monitor_process(self):
        while True:
            now = self.env.now
            if now >= self.warmup:
                self.sw[now] = self.social_welfare()
                self.demandstats[now] = self.demand_stats()
                X = self.cachesToMatrix()
                ecc = self.expected_caching_cost(X)
                tot = self.cost_without_caching()
                ecg = tot - ecc
                self.funstats[now] = (ecc, tot, ecg)
                logging.info(pp(
                    [now, ':', 'Utility = %f' % self.sw[now], ', DEMSTATS =', self.demandstats[now],
                     'ECC = %f, TOT = %f, ECG = %f' % self.funstats[now] ]))

            yield self.env.timeout(random.expovariate(self.monitoring_rate))


class NetworkedCache(object):
    """An abstract networked cache.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, capacity, _id):
        pass

    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def perm_set(self):
        pass

    @abstractmethod
    def makePermanent(self, item):
        pass

    @abstractmethod
    def receive(self, message, edge, time):
        pass

    @abstractmethod
    def __contains__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def isPermanent(self, item):
        pass


class PriorityNetworkCache(NetworkedCache):
    """ A Priority Networked Cache. Supports LRU,LFU, and RR policies.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the LRU principle.
    """

    def __init__(self, capacity, _id, principle):
        self.cache = PriorityCache(capacity, _id)
        self.permanent_set = set([])
        self._id = _id
        self._capacity = capacity
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.principle = principle

    def __str__(self):
        return str(self.cache) + '+' + str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def isPermanent(self, item):
        return item in self.permanent_set

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def payload(self, contents):
        for i in range(len(contents)) :
            contents[i][0] = "RESPONSE"
        return contents

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            item = d.item
            logging.debug(pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            inside_cache = item in self.cache
            inside_permanent_set = item in self.permanent_set
            # if this node caches item
            if inside_cache or inside_permanent_set:
                logging.debug(pp(
                    [now, ': Item', item, 'is inside', 'permanent set' if inside_permanent_set else 'cache', 'of',
                     self._id]))
                if inside_cache:  # i.e., not in permanent set
                    princ_map = {'LRU': now, 'LFU': self.cache.priority(item) + 1, 'RR': random.random(),
                                 'FIFO': self.cache.priority(item)}
                    logging.debug(
                        pp([now, ': Priority of', item, 'updated to', princ_map[self.principle], 'at cache', self._id]))
                    self.cache.add(item, princ_map[self.principle])
                self.stats['hits'] += 1
                # if this node is query node
                if self._id == d.query_source:
                    logging.debug(pp(
                        [now, ': Response to query', query_id, 'of', d, 'delivered to query source by cache',
                         self._id]))
                    pred = d  # this demand object
                else:
                    pred = d.pred(self._id)  # previous node in path
                    logging.debug(pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                # a response is generate
                payload_rmsg = self.payload(msg.payload)
                rmsg = ResponseMessage(d, query_id, payload_rmsg, stats=msg.stats)
                return [(rmsg, e)]
            # if this node does not cache item
            else:
                logging.debug(pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                # a tuple for this node and next node in path
                e = (self._id, succ)
                return [(msg, e)]

        if label == "RESPONSE":
            logging.debug(pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            princ_map = {'LRU': now, 'LFU': self.cache.priority(item) + 1 if item in self.cache else 1,
                         'RR': random.random(), 'FIFO': self.cache.priority(item) if item in self.cache else now}
            logging.debug(pp([now, ': Priority of', item, 'updated to', princ_map[self.principle], 'at', self._id]))
            self.cache.add(item, princ_map[self.principle])  # add the item to the cache/update priority
            # if this node is query node
            if d.query_source == self._id:
                logging.debug(
                    pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                pred = d  # this demand object
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d, 'passes through cache', self._id,
                                  'moving further down path']))
                pred = d.pred(self._id)  # previous node in path
            e = (self._id, pred)
            return [(msg, e)]


class CGCache(NetworkedCache):
    """ A Priority Networked Cache. Supports LRU,LFU, and RR policies.

    Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the LRU principle.
    """

    def __init__(self, capacity, _id, X):
        self.cache = np.nonzero(X)[0]  # get the item stored in node
        self.permanent_set = set([])
        self._id = _id
        self._capacity = capacity
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0

    def __str__(self):
        return str(self.cache) + '+' + str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def isPermanent(self, item):
        return item in self.permanent_set

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def payload(self, contents):
        for i in range(len(contents)) :
            contents[i][0] = "RESPONSE"
        return contents

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            item = d.item
            logging.debug(pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            inside_cache = item in self.cache
            inside_permanent_set = item in self.permanent_set
            # if this node caches item
            if inside_cache or inside_permanent_set:
                logging.debug(pp(
                    [now, ': Item', item, 'is inside', 'permanent set' if inside_permanent_set else 'cache', 'of',
                     self._id]))

                self.stats['hits'] += 1
                # if this node is query node
                if self._id == d.query_source:
                    logging.debug(pp(
                        [now, ': Response to query', query_id, 'of', d, 'delivered to query source by cache',
                         self._id]))
                    pred = d  # this demand object
                else:
                    pred = d.pred(self._id)  # previous node in path
                    logging.debug(pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                # a response is generate
                payload_rmsg = self.payload(msg.payload)
                rmsg = ResponseMessage(d, query_id, payload_rmsg, stats=msg.stats)
                return [(rmsg, e)]
            # if this node does not cache item
            else:
                logging.debug(pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                # a tuple for this node and next node in path
                e = (self._id, succ)
                return [(msg, e)]

        if label == "RESPONSE":
            logging.debug(pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item

            # if this node is query node
            if d.query_source == self._id:
                logging.debug(
                    pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                pred = d  # this demand object
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d, 'passes through cache', self._id,
                                  'moving further down path']))
                pred = d.pred(self._id)  # previous node in path
            e = (self._id, pred)
            return [(msg, e)]


def main():
    # logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Simulate a Network of Caches',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('inputfile',help = 'Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.')
    parser.add_argument('Graph_instance', help='Graph instance')
    parser.add_argument('problem_instance', help='Problem instance')
    parser.add_argument('integer_solution', help='rounding result')
    parser.add_argument('outputfile', help='Output file')
    parser.add_argument('--queue_type', default='MMInfQueue', type=str, help='Queue type', choices=['MMInfQueue', 'CountQueue'])
    parser.add_argument('--time', default=1000.0, type=float, help='Total simulation duration')
    parser.add_argument('--warmup', default=0.0, type=float, help='Warmup time until measurements start')
    parser.add_argument('--random_seed', default=4156910908, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--cache_type', default='LRU', type=str, help='Networked Cache type',
                        choices=['LRU', 'FIFO', 'LFU', 'RR', 'CG'])
    parser.add_argument('--query_message_length', default=0.0, type=float, help='Query message length')
    parser.add_argument('--response_message_length', default=0.0, type=float, help='Response message length')
    parser.add_argument('--monitoring_rate', default=1.0, type=float, help='Monitoring rate')


    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    
    def cacheGenerator(capacity, _id, X):
        if args.cache_type == 'LRU':
            return PriorityNetworkCache(capacity, _id, 'LRU')
        if args.cache_type == 'LFU':
            return PriorityNetworkCache(capacity, _id, 'LFU')
        if args.cache_type == 'FIFO':
            return PriorityNetworkCache(capacity, _id, 'FIFO')
        if args.cache_type == 'RR':
            return PriorityNetworkCache(capacity, _id, 'RR')
        if args.cache_type == 'CG':
            return CGCache(capacity, _id, X)

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2015)

    CONFIG.QUERY_MESSAGE_LENGTH = args.query_message_length
    CONFIG.RESPONSE_MESSAGE_LENGTH = args.response_message_length

    input_graph = "INPUT/" + args.Graph_instance
    input_problem = "INPUT_NEW/"+ args.problem_instance
    integer_solution = "ROUNDED/" + args.integer_solution
    P = Problem.unpickle_cls(input_problem)
    with file(input_graph, 'r') as f:
        G = pickle.load(f)
    l = np.load(integer_solution+'.npy')
    X = l[1]

    demands = P.demands
    item_sources = P.item_sources
    capacities = P.capacities
    weights = P.EDGE
    utilityfunction_rho = P.utilityfunction

    def utilityfunction_n(x):
        return x**2

    if args.queue_type == 'CountQueue':
        Queuing_type = CountQueue
    elif args.queue_type == 'MMInfQueue':
        Queuing_type = MMInfQueue
    else:
        Queuing_type = None

    logging.info('Building CacheNetwork')
    cnx = CacheNetwork(G, cacheGenerator, demands, item_sources, capacities, weights, weights, utilityfunction_n, utilityfunction_rho, X, Queuing_type, args.warmup,
                       args.monitoring_rate)
    logging.info('...done')


    cnx.run(args.time)

    demand_stats = {}
    node_stats = {}
    network_stats = {}

    for d in cnx.demands:
        demand_stats[str(d)] = cnx.demands[d]['stats']
        demand_stats[str(d)]['queries_spawned'] = cnx.demands[d]['queries_spawned']
        demand_stats[str(d)]['queries_satisfied'] = cnx.demands[d]['queries_satisfied']

    for x in cnx.nodes():
        node_stats[x] = cnx.node[x]['cache'].stats

    network_stats['demand'] = cnx.demandstats
    network_stats['fun'] = cnx.funstats
    network_stats['utility'] = cnx.sw
    Time_average = sum(network_stats['utility'].values())/len(network_stats['utility'])
    print Time_average


    times1 = sorted(network_stats['fun'].keys())
    times2 = sorted(network_stats['utility'].keys())
    eccs = [network_stats['fun'][t][0] for t in times1]

    tags = [network_stats['utility'][t] for t in times2]

    fig, ax = plt.subplots()
    names = ['ECC', 'TACC']
    forms = ['k:', 'rs-', 'b^-']
    ax.plot(times1, eccs, forms[1])
    ax.plot(times2, tags, forms[2])
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Cost', fontsize=15)
    ax.legend(names)
    plt.title(args.cache_type)

    plt.show()

    dir = "LRU/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    out = dir + args.outputfile + "%s_%s_%ftime" % (args.graph_instance, args.cache_type, args.time)

    with open(out, 'wb') as f:
        pickle.dump([args, demand_stats, node_stats, network_stats], f)


if __name__ == "__main__":
    main()
