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
        self.payload = [payload]
        self.length = length
        self.u_bit = u_bit
        self.counter = counter # record how many requests are merged into one
        if stats == None:
            self.stats = {}
            self.stats['delay'] = 0.0
            self.stats['hops'] = 0.0
            self.stats['weight'] = 0.0
            self.stats['downweight'] = 0.0
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
        Message.__init__(self, header=("QUERY", d, query_id), payload=("QUERY", d, query_id), length=CONFIG.QUERY_MESSAGE_LENGTH,
                         stats=stats, u_bit=True)


class ResponseMessage(Message):
    """
     A response message.
    """

    def __init__(self, d, query_id, stats=None):
        Message.__init__(self, header=("RESPONSE", d, query_id), payload=("QUERY", d, query_id), length=CONFIG.RESPONSE_MESSAGE_LENGTH,
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

    def __init__(self, G, cacheGenerator, demands, item_sources, capacities, weights, delays, utilityfunction ,warmup=0,
                 monitoring_rate=1.0):
        self.env = Environment()
        self.warmup = warmup
        self.demandstats = {}
        self.sw = {}
        self.funstats = {}
        self.optstats = {}
        self.monitoring_rate = monitoring_rate
        self.utilityfunction = utilityfunction

        DiGraph.__init__(self, G)
        for x in self.nodes():
            self.node[x]['cache'] = cacheGenerator(capacities[x], x)
            self.node[x]['pipe'] = Store(self.env)
        for e in self.edges():
            x = e[0]
            y = e[1]
            reverse_e = (y,x)
            if e in delays:
                self.edge[x][y]['weight'] = weights[e]
                self.edge[x][y]['delay'] = delays[e].copy()
                for demand in self.edge[x][y]['delay']:
                    self.edge[x][y]['delay'][demand] = float('inf')
                self.edge[x][y]['pipe_up'] = MultiQueue(self.env, MMInfQueue, self.edge[x][y]['delay'], self.node[y]['pipe'])
            elif reverse_e in delays:
                self.edge[x][y]['weight'] = weights[reverse_e]
                self.edge[x][y]['delay'] = delays[reverse_e].copy
                self.edge[x][y]['pipe_down'] = MultiQueue(self.env, MMInfQueue, self.edge[x][y]['delay'], self.node[y]['pipe'])

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
                #  msg.stats:'delay''hops''weight''downweight'
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
            (msg, e) = yield self.node[x]['pipe'].get() # e: (demand, demand.query_source)
            generated_messages = self.node[e[1]]['cache'].receive(msg, e,
                                                                  self.env.now)  # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!!
            for (new_msg, new_e) in generated_messages:
                if new_e[1] in self.demands: # this message is sent to its query node
                    yield self.demands[new_e[1]]['pipe'].put(new_msg) 
                else:
                    label, d, query_id = new_msg.header
                    if label == "QUERY":
                        self.edge[new_e[0]][new_e[1]]['pipe_up'].enqueue(new_msg, d)
                    elif label == "RESPONSE":
                        self.edge[new_e[0]][new_e[1]]['pipe_down'].enqueue(new_msg, d)

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

    def statesToMatrix(self):
        """Constructs a matrix containing marginal information. This assumes that caches contain a state() function, capturing maginals. Only LMin implements this
        """
        zipped = []
        n = len(self.nodes())
        m = max([d.item for d in self.demands]) + 1
        Y = matrix()
        for x in self.nodes():
            for item in self.node[x]['cache'].non_zero_state_items():
                zipped.append((self.node[x]['cache'].state(item), x, item))

        val, I, J = zip(*zipped)

        return spmatrix(val, I, J, size=(n, m)) # a node*item matrix 

    def social_welfare(self):
        """ Function computing the social welfare.
        """
        utility = 0.0
        for e in self.edge():
            x = e[0]
            y = e[1]
            if self.edge[x][y].has_key('pipe_down'):
                utility += self.utilityfunction(self.edge[x][y]['pipe_down'].get_custormers())
        return utility

        dsw = 0.0
        hsw = 0.0
        wsw = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate
            x = d.query_source
            while not (item in self.node[x][
                'cache'] or x is d.item_source):  # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!
                s = d.succ(x)
                dsw += rate * (CONFIG.QUERY_MESSAGE_LENGTH * self.edge[x][s]['delay'] + CONFIG.RESPONSE_MESSAGE_LENGTH *
                               self.edge[s][x]['delay'])
                wsw += rate * (self.edge[x][s]['weight'] + self.edge[s][x]['weight'])
                hsw += rate * (2)
                x = s

        return (dsw / sumrate, hsw / sumrate, wsw / sumrate)

    def caching_gain(self):
        """ Function computing the caching gain under the present caching situation
        """
        X = self.cachesToMatrix()
        return self.expected_caching_gain(X)

    def cost_without_caching(self):
        """ Function computing the  cost of recovering all items demanded from respective sources."""
        cost = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            while s is not None:
                cost += rate * (self.edge[s][x]['weight'] + self.edge[x][s]['weight'])
                x = s
                s = d.succ(x)

        return cost / sumrate

    def expected_caching_gain(self, Y):
        """ Function computing the expected caching gain under marginals Y, presuming product form. Also computes deterministic caching gain if Y is integral.
        """
        ecg = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            prodsofar = 1 - Y[x, item]
            while s is not None:
                ecg += rate * (self.edge[s][x]['weight'] + self.edge[x][s]['weight']) * (1 - prodsofar)

                x = s
                s = d.succ(x)
                prodsofar *= 1 - Y[x, item]

        return ecg / sumrate

    def relaxation(self, Y):
        """ Function computing the relaxation of caching gain under marginals Y. Relaxation equals deterministic caching gain if Y is integral.
        """
        rel = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            sumsofar = Y[x, item]
            while s is not None:
                rel += rate * (self.edge[s][x]['weight'] + self.edge[x][s]['weight']) * min(1.0, sumsofar)

                x = s
                s = d.succ(x)
                sumsofar += Y[x, item]

        return rel / sumrate

    def demand_stats(self):
        """ Computed stats across demands.
        """
        stats = {}
        queries_logged = 0.0
        rate = 0.0
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
                ecg = self.expected_caching_gain(X)
                rel = self.relaxation(X)
                tot = self.cost_without_caching()
                esw = tot - ecg
                self.funstats[now] = (ecg, rel, esw, tot)
                logging.info(pp(
                    [now, ':', 'DSW = %f, HSW = %f, WSW = %f' % self.sw[now], ', DEMSTATS =', self.demandstats[now],
                     'FUNSTATS =', self.funstats[now]]))
                try:
                    # if True:
                    Y = self.statesToMatrix()
                    secg = self.expected_caching_gain(Y)
                    srel = self.relaxation(Y)
                    self.funstats[now] += (secg, srel)
                    logging.info(pp([now, ': SECG=', secg, 'SREL=', srel]))
                except AttributeError:
                    logging.debug(pp([now, ": No states in this class"]))

            yield self.env.timeout(random.expovariate(self.monitoring_rate))

    def minimizeRelaxation(self):
        n = len(self.nodes())
        m = max([d.item for d in self.demands]) + 1
        number_of_placement_variables = n * m

        def position(node, item):
            return node * m + item

        A = []
        b = []
        row = 0

        # Permanent set constraints
        logging.debug('Creating permanent set constaints...')
        row = 0
        for x in self.nodes():
            perm_set = self.node[x]['cache'].perm_set()
            if len(perm_set) > 0:
                for item in perm_set:
                    A += [(1.0, row, position(x, item))]
                    b += [1.0]
                    row += 1
        logging.debug('...done. Created %d constraints' % row)

        total_equality_constraints = row

        G = []
        h = []

        # Capacity constraints
        logging.debug('Creating capacity constaints...')
        G += [(1.0, x, position(x, item)) for item in xrange(m) for x in self.nodes()]
        h += [self.node[x]['cache'].capacity() + len(self.node[x]['cache'].perm_set()) for x in self.nodes()]
        # h += [ self.node[x]['cache'].capacity()    for x in self.nodes()]
        logging.debug('...done at %d rows' % len(h))

        row = n
        t = number_of_placement_variables

        # t's smaller than sums of y_vi's in path
        logging.debug('Creating t up constraints...')
        for d in self.demands:
            item = d.item
            path = d.path
            sofar = []
            for v in path:
                if len(sofar) > 0:
                    for u in sofar:
                        G += [(-1.0, row, position(u, item))]
                    G += [(1.0, row, t)]
                    h += [0.0]
                    row += 1
                    t += 1
                sofar += [v]

        logging.debug('...done at %d rows' % row)

        total_ts = t - number_of_placement_variables

        # t's smaller than 1.0
        logging.debug('Creating t ll one constraints...')
        t = number_of_placement_variables
        while t < (number_of_placement_variables + total_ts):
            G += [(1.0, row, t)]
            h += [1.0]
            row += 1
            t += 1

        logging.debug('...done at %d rows' % row)

        # y's less than 1
        logging.debug('Creating y ll one gg 0 constraints...')
        y = 0
        while y < number_of_placement_variables:
            G += [(1.0, row, y)]
            h += [1.0]
            row += 1
            G += [(-1.0, row, y)]
            h += [0.0]
            row += 1
            y += 1
        logging.debug('...done at %d rows' % row)

        total_inequality_constraints = row

        # objective
        logging.debug('Creating objective vector...')
        c = number_of_placement_variables * [0]
        for d in self.demands:
            rate = d.rate
            path = d.path

            x = d.query_source
            s = d.succ(x)
            while s is not None:
                c += [-rate * (self.edge[s][x]['weight'] + self.edge[x][s]['weight'])]
                x = s
                s = d.succ(x)

        logging.debug('...done at %d terms' % len(c))

        val, I, J = zip(*A)
        A = spmatrix(val, I, J, size=(total_equality_constraints, number_of_placement_variables + total_ts))
        b = matrix(b)

        val, I, J = zip(*G)
        G = spmatrix(val, I, J, size=(total_inequality_constraints, number_of_placement_variables + total_ts))
        h = matrix(h)

        c = matrix(c)

        logging.debug('c has length %d ' % len(c))
        logging.debug('G has dims %d x %d and matrix_rank ' % G.size + str(matrix_rank(G)))
        logging.debug('h has length %d ' % len(h))
        logging.debug('A has dims %d x %d and matrix_rank' % A.size + str(matrix_rank(A)))
        logging.debug('b has length %d ' % len(b))

        res = lp(c, G, h, A,
                 b)  # , primalstart={'x':matrix(np.zeros(c.size)*1.e-99),'s':matrix( total_inequality_constraints*[1.e-20])})

        opt = res['x'][:number_of_placement_variables]
        return np.reshape(opt, (n, m), order='C'), res


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
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)
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


class ExploreMessage(Message):
    """
     An exploration message.
    """

    def __init__(self, d, query_id, initiator, stats=None):
        Message.__init__(self, header=("EXPLORE", d, query_id), payload=None, length=CONFIG.EXPLORE_MESSAGE_LENGTH,
                         stats=stats, u_bit=True)
        self.explore_source = initiator


class ExploreResponseMessage(Message):
    """
     An exploration response message.
    """

    def __init__(self, d, query_id, initiator, stats=None):
        Message.__init__(self, header=("EXPLORE_RESPONSE", d, query_id), payload=None,
                         length=CONFIG.EXPLORE_MESSAGE_LENGTH, stats=stats, u_bit=False)
        self.explore_source = initiator


class EWMAGradCache(NetworkedCache):
    """ An EWMA Gradient Networked Cache.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

    def __init__(self, capacity, _id, beta=1.0):
        self.cache = EWMACache(capacity, _id, beta)
        self._id = _id
        self.permanent_set = set([])
        self._capacity = capacity
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.stats['explores'] = 0.0
        self.stats['explore_responses'] = 0.0

    def __str__(self):
        return 'Cache: ' + str(self.cache) + 'Permanent: ' + str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def isPermanent(self, item):
        return item in self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            item = d.item
            logging.debug(pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            inside = item in self.cache or item in self.permanent_set
            # if this node caches this item
            if inside:
                logging.debug(pp([now, ': Item', item, 'is inside ', self._id]))
                self.stats['hits'] += 1
                # if this node is query node
                if self._id == d.query_source:
                    logging.debug(
                        pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                    pred = d
                else:
                    pred = d.pred(self._id)
                    logging.debug(pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                # generate a response
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)

                msglist = [(rmsg, e)]
                # if this node is not the designated source 
                if item not in self.permanent_set:
                    succ = d.succ(self._id)
                    # if this node has next node on its path
                    if succ != None:
                        logging.debug(pp([now, ': Item', item, 'is inside cache of ', self._id,
                                          ' will forward exploration message']))
                        # A exploreMessage will go along this path still
                        emsg = ExploreMessage(d, query_id, self._id)
                        e = (self._id, succ)
                        msglist += [(emsg, e)]

                return msglist
            else:
                logging.debug(pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                return [(msg, e)]

        if label == "RESPONSE":
            logging.debug(pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            logging.debug(pp([now, ': Node', self._id, 'updating derivative of item', item, 'with measurement',
                              msg.stats['downweight']]))
            self.cache.add(item, msg.stats['downweight'],
                           now)  # does not really add item, but updates its gradient estimate
            logging.debug(pp([now, ': Node', self._id, ' now stores', self.cache]))

            if d.query_source == self._id:
                logging.debug(
                    pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                pred = d
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d, 'passes through cache', self._id,
                                  'moving further down path']))
                pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]

        if label == "EXPLORE":
            item = d.item
            logging.debug(pp([now, ': Explore message for item', item, 'received by cache', self._id]))
            self.stats['explores'] += 1.0

            inside = item in self.cache or item in self.permanent_set
            if inside:
                logging.debug(pp([now, ': Item', item, 'is inside ', self._id]))
                pred = d.pred(self._id)
                logging.debug(
                    pp([now, ': Explore Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                ermsg = ExploreResponseMessage(d, query_id, msg.explore_source, stats=msg.stats)

                return [(ermsg, e)]

            else:
                logging.debug(pp([now, ': Item', item, 'is not inside', self._id, 'continue exploring']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.debug(pp([now, ':Exploration for', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                return [(msg, e)]

        if label == "EXPLORE_RESPONSE":
            logging.debug(pp([now, ': Explore Response message for', d, 'received by cache', self._id]))
            self.stats['explore_responses'] += 1.0
            item = d.item

            if msg.explore_source == self._id:
                logging.debug(
                    pp([now, ': Node', self._id, 'received final exploration response for', query_id, 'of', d]))
                logging.debug(pp([now, ': Node', self._id, 'updating derivative of item', item, 'with measurement',
                                  msg.stats['downweight']]))
                self.cache.add(item, msg.stats['downweight'],
                               now)  # does not really add item, but updates its gradient estimate
                logging.debug(pp([now, ': Node', self._id, ' now stores', self.cache]))
                return []

            logging.debug(pp([now, ': Explore Response to query', query_id, 'of', d, 'passes through cache', self._id,
                              'moving further down path']))
            pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]


class LMinCache(NetworkedCache):
    """ A Networked Cache minimizing the relaxation L.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

    def __init__(self, capacity, _id, gamma=0.1, T=5, expon=0.5, interpolate=False):
        self.cache = LMinimalCache(capacity, _id, gamma, expon, interpolate)
        self._id = _id
        self.permanent_set = set([])
        self._capacity = capacity
        self.T = T
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.stats['explores'] = 0.0
        self.stats['explore_responses'] = 0.0

    def __str__(self):
        return 'Cache: ' + str(self.cache) + 'Permanent: ' + str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def isPermanent(self, item):
        return item in self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            msglist = []
            item = d.item

            logging.debug(pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            if self._id == d.query_source:
                succ = d.succ(self._id)
                if succ != None:
                    logging.debug(pp([now, ': Item', item, 'is inside cache of ', self._id,
                                      ' which is the query source, will prepare an exploration message']))
                    emsg = ExploreMessage(d, query_id, self._id)
                    e = (self._id, succ)
                    sumsofar = self.cache.state(item) + float(item in self.permanent_set)
                    emsg.payload = sumsofar
                    if sumsofar < 1.0:
                        msglist.append((emsg, e))

            inside_cache = item in self.cache
            inside_perm = item in self.permanent_set
            if inside_cache or inside_perm:
                logging.debug(pp([now, ': Item', item,
                                  'is inside %s of %d' % ('cache' if inside_cache else 'permanent set', self._id)]))
                self.stats['hits'] += 1
                if self._id == d.query_source:
                    logging.debug(
                        pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                    pred = d
                else:
                    pred = d.pred(self._id)
                    logging.debug(pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)

                msglist.append((rmsg, e))

            else:
                logging.debug(pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                msglist.append((msg, e))

            return msglist

        if label == "RESPONSE":
            logging.debug(pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            # logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
            # self.cache.add(item,msg.stats['downweight'],now) #does not really add item, but updates its gradient estimate
            # logging.debug(pp([now,': Node',self._id,' now stores',self]))

            if d.query_source == self._id:
                logging.debug(
                    pp([now, ': Response to query', query_id, 'of', d, ' finally delivered by cache', self._id]))
                pred = d
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d, 'passes through cache', self._id,
                                  'moving further down path']))
                pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]

        if label == "EXPLORE":
            item = d.item
            logging.debug(pp([now, ': Explore message for item', item, 'received by cache', self._id,
                              'sumsofar is %f local state is %f in permanent is %f' % (
                              msg.payload, self.cache.state(item), float(item in self.permanent_set))]))
            self.stats['explores'] += 1.0

            new_sum = msg.payload + self.cache.state(item) + float(item in self.permanent_set)
            if new_sum > 1.0 or item in self.permanent_set:
                logging.debug(pp(
                    [now, ': Item', item, 'has aggregate state value', new_sum, 'on node', self._id, 'of demand', d]))
                pred = d.pred(self._id)
                logging.debug(
                    pp([now, ': Explore Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                ermsg = ExploreResponseMessage(d, query_id, msg.explore_source, stats=msg.stats)

                return [(ermsg, e)]

            else:
                logging.debug(
                    pp([now, ': Item', item, 'is has agreggate', new_sum, 'at node', self._id, 'continue exploring']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Exploration for', query_id, 'of', d, 'reached', self._id,
                                      'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                msg.payload = new_sum
                return [(msg, e)]

        if label == "EXPLORE_RESPONSE":
            logging.debug(pp([now, ': Explore Response message for', d, 'received by cache', self._id]))
            self.stats['explore_responses'] += 1.0
            item = d.item

            logging.debug(pp([now, ': Node', self._id, 'received exploration response for', query_id, 'of', d]))
            logging.debug(pp([now, ': Node', self._id, 'updating derivative of item', item, 'with measurement',
                              msg.stats['downweight']]))
            self.cache.updateGradient(item, msg.stats['downweight'])

            if msg.explore_source == self._id:
                logging.debug(
                    pp([now, ': Node', self._id, 'is terminal node for exploration response for', query_id, 'of', d]))
                # logging.debug(pp([now,': Node',self._id,' now stores',self.cache]))
                return []

            logging.debug(pp([now, ': Explore Response to query', query_id, 'of', d, 'passes through cache', self._id,
                              'moving further down path']))
            pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]

    def shuffleProcess(self):
        """A process handling messages sent to caches.

           It is effectively a wrapper for a receive call, made to a NetworkedCache object.

        """
        while True:
            logging.debug(pp([self.env.now, ': New suffling of cache', self._id]))
            # key,placements,probs,distr=self.cache.shuffle(self.env.now)
            self.cache.shuffle(self.env.now)
            logging.debug(pp([self.env.now, ': New cache at', self._id, ' is', self.cache]))
            # logging.debug('Setting cache at %d to %s, probability was:%f' %(self._id,str(self.cache),probs[key]) )

            yield self.env.timeout(self.T)

    def startShuffleProcess(self, env):
        self.env = env
        self.env.process(self.shuffleProcess())

    def state(self, item):
        return max(self.cache.state(item), float(item in self.permanent_set))

    def non_zero_state_items(self):
        return self.cache._state.keys() + list(self.permanent_set)


def main():
    # logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Simulate a Network of Caches',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('inputfile',help = 'Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.')
    parser.add_argument('Graph_instance', help='Graph instance')
    parser.add_argument('problem_instance', help='Problem instance')
    parser.add_argument('outputfile', help='Output file')

    parser.add_argument('--time', default=1000.0, type=float, help='Total simulation duration')
    parser.add_argument('--warmup', default=0.0, type=float, help='Warmup time until measurements start')
    parser.add_argument('--random_seed', default=4156910908, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--cache_type', default='LRU', type=str, help='Networked Cache type',
                        choices=['LRU', 'FIFO', 'LFU', 'RR', 'EWMAGRAD', 'LMIN'])
    #   parser.add_argument('--cache_keyword_parameters',default='{}',type=str,help='Networked Cache additional constructor parameters')
    parser.add_argument('--query_message_length', default=0.0, type=float, help='Query message length')
    parser.add_argument('--response_message_length', default=0.0, type=float, help='Response message length')
    parser.add_argument('--monitoring_rate', default=1.0, type=float, help='Monitoring rate')
    parser.add_argument('--interpolate', default=False, type=bool, help='Interpolate past states, used by LMIN')
    parser.add_argument('--beta', default=1.0, type=float, help='beta used in EWMA')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma used in LMIN')
    parser.add_argument('--expon', default=0.5, type=float, help='exponent used in LMIN')
    parser.add_argument('--T', default=5., type=float, help='Suffling period used in LMIN')
    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    
    def cacheGenerator(capacity, _id):
        if args.cache_type == 'LRU':
            return PriorityNetworkCache(capacity, _id, 'LRU')
        if args.cache_type == 'LFU':
            return PriorityNetworkCache(capacity, _id, 'LFU')
        if args.cache_type == 'FIFO':
            return PriorityNetworkCache(capacity, _id, 'FIFO')
        if args.cache_type == 'RR':
            return PriorityNetworkCache(capacity, _id, 'RR')
        if args.cache_type == 'EWMAGRAD':
            return EWMAGradCache(capacity, _id, beta=args.beta)
        if args.cache_type == 'LMIN':
            return LMinCache(capacity, _id, gamma=args.beta, T=args.T, expon=args.expon, interpolate=args.interpolate)

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed + 2015)

    CONFIG.QUERY_MESSAGE_LENGTH = args.query_message_length
    CONFIG.RESPONSE_MESSAGE_LENGTH = args.response_message_length

    input_graph = "INPUT/" + args.Graph_instance
    input_problem = "INPUT_NEW/"+ args.problem_instance
    P = Problem.unpickle_cls(input_problem)
    with file(input_graph, 'r') as f:
        G = pickle.load(f)
    f.close()
    
    demands = P.demands
    item_sources = P.item_sources
    capacities = P.capacities
    weights = P.EDGE

    logging.info('Building CacheNetwork')
    cnx = CacheNetwork(G, cacheGenerator, demands, item_sources, capacities, weights, weights, utilityfunction, args.warmup,
                       args.monitoring_rate)
    logging.info('...done')

#    Y, res = cnx.minimizeRelaxation()

#    logging.info('Optimal Relaxation is: ' + str(cnx.relaxation(Y)))
#    logging.info('Expected caching gain at relaxation point is: ' + str(cnx.expected_caching_gain(Y)))

#    optimal_stats = {}
#    optimal_stats['res'] = res
#    optimal_stats['Y'] = Y
#    optimal_stats['L'] = cnx.relaxation(Y)
#    optimal_stats['F'] = cnx.expected_caching_gain(Y)

    if args.cache_type == "LMIN":
        for x in cnx.nodes():
            cnx.node[x]['cache'].startShuffleProcess(cnx.env)

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
    network_stats['opt'] = cnx.optstats

    out = args.outputfile + "%s_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%ftime_%fgamma_%fexpon%fbeta" % (
    args.graph_type, args.cache_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.time,
    args.gamma, args.expon, args.beta)

    with  open(out, 'wb') as f:
        pickle.dump([args, demand_stats, node_stats, network_stats], f)


#   for d in cnx.demands:
#	print d.item, d.rate, d.requests_tally.count()/time, len(d.path), d.hops_tally.mean(), d.weight_tally.mean(), d.time_tally.mean(), d.hit_source_tally.mean()

#   for x in cnx.nodes():
#	cache = cnx.node[x]['cache']
#	print x,cache.queries_tally.count(), cache.hits_tally.mean(), cache.downloads_tally.count()/time

# def plot_ecdf(y,x_label):
#     ecdf = ECDF(y)
#     x= sorted(list(set(y)))
#     plt.plot(x,ecdf(x))
#     plt.xlabel(x_label)
#     plt.ylabel('CDF')
#     plt.show()

if __name__ == "__main__":
    main()
