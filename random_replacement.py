from helpers import constructDistribution,write
import time
from Toplogy_gen  import Problem, Demand
import argparse
from cvxopt import matrix
import random
import os


def OBJ_dict(P, ro_dict):
    obj = 0.

    for edge in ro_dict:
        for demand in ro_dict[edge]:
            ro = ro_dict[edge][demand]
            obj = obj+P.utilityfunction(ro,0)
    return obj

def ro_uvr(P, X):
    """Compute  rho for each edge and demand, given a integer solution X."""
    ro_uvr = {}
    # Go through demands
    for demand in P.demands:
        path = demand.path
        item = demand.item
        rate = demand.rate

        for node_i in range(1,len(path)):
            x = X[path[node_i-1],item]
            if x == 0:
                edge = (path[node_i-1],path[node_i])
                ro = rate/P.EDGE[edge][demand]
            else:
                break
            if edge in ro_uvr:
                ro_uvr[edge][demand] = ro
            else:
                ro_uvr[edge]={demand: ro}
    return ro_uvr


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Simulate the Random Placement Alg.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--order_moment', default=2, type=int, help='Order of moment for the expected cost', choices=[1,2,3,4])
    parser.add_argument('--queue_type', default='MMInfty', type=str, help='Type of queue', choices=['MMInfty', 'Info'])

    parser.add_argument('--assign_order', default="service_first", type=str, help='the order of assigning items and service rate',
                        choices=['item_first', 'service_first'])
    args  = parser.parse_args()
    problem_instance = "problem_" + args.graph_type + "_1000demands_100catalog_size_2mincap_2maxcap_100size_powerlaw_rate1.0_" + str(args.query_nodes) + "qnodes_" + str(args.order_moment) + "order_" + args.queue_type
    input = "INPUT/"+problem_instance
    P = Problem.unpickle_cls(input)
    random.seed(1993)
    capacities = P.capacities
    dem_items=[]
    for demand in P.demands:
         item = demand.item
         dem_items.append(item)
    V,I = (len(capacities),len(set(dem_items)))

    track = []
    X = matrix(0, (V, I))
    ro_uv_dic = ro_uvr(P, X)
    OBJ0 = OBJ_dict(P, ro_uv_dic)
    P.cost = OBJ0
    track.append(OBJ0)
    num_sample = 10

    '''Place item randomly, then assign service rate uniformly according to placement'''
    def item_first(P, OBJ0, num_sample):
        sum_time = 0.0
        sum_obj = 0.0
        for samp in range(num_sample):
            X = matrix(0, (V, I))
            tstart = time.time()
            for v in range(V):
    #            d = {}
    #            for i in range(I):
    #                d[i] = capacities[v]*1./I # probability of item i being cached
    #            if capacities[v]>0:
    #                placements, probs, distr = constructDistribution(d,capacities[v])
    #                key = distr.rvs(size =1)
    #                cache_v = placements[key[0]]
    #                for i in cache_v:
    #                    X[v,i] = 1
                cache_v = []
                '''randomly choose items for each node'''
                while len(set(cache_v)) < capacities[v]:
                    cache_v += random.sample(set(dem_items), 1)
                for i in cache_v:
                    X[v, i] = 1
            ro_uv_dic = ro_uvr(P, X)

            '''optimize over mu'''
            for edge in ro_uv_dic:
                num_demand = len(ro_uv_dic[edge])
                new_mu = P.capacity_servicerate[
                             edge] / num_demand  # assign uniform service rate to each request for each edge
                for demand in ro_uv_dic[edge]:
                    ro_uv_dic[edge][demand] = ro_uv_dic[edge][demand] * P.EDGE[edge][demand] / (new_mu)

            elapsed = time.time() - tstart
            OBJ = OBJ_dict(P, ro_uv_dic)
            sum_obj += OBJ0 - OBJ
            sum_time += elapsed
        return sum_time/num_sample, sum_obj/num_sample

    '''Assign service rate according to request, then '''
    def service_first(P, OBJ0, num_sample):
        sum_time = 0.0
        sum_obj = 0.0
        '''optimize for mu'''
        for edge in P.EDGE:
            num_demand = len(P.EDGE[edge])
            new_mu = P.capacity_servicerate[edge] / num_demand
            for demand in P.EDGE[edge]:
                P.EDGE[edge][demand] = new_mu

        for sample in range(num_sample):
            X = matrix(0, (V, I))
            tstart = time.time()
            for v in range(V):
                cache_v = []
                '''randomly choose items for each node'''
                while len(set(cache_v)) < capacities[v]:
                    cache_v += random.sample(set(dem_items), 1)
                for i in cache_v:
                    X[v, i] = 1
            ro_uv_dic = ro_uvr(P, X)

            elapsed = time.time() - tstart
            OBJ = OBJ_dict(P, ro_uv_dic)
            sum_obj += OBJ0 - OBJ
            sum_time += elapsed
        return sum_time/num_sample, sum_obj/num_sample


    if args.assign_order == 'item_first':
        [ave_time, ave_obj] = item_first(P, OBJ0, num_sample)
    else:
        [ave_time, ave_obj] = service_first(P, OBJ0, num_sample)

    track.append((ave_time,ave_obj))
    print track
    dir_output = "RANDOM/"
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    outputfile = problem_instance + "-" + args.assign_order
    output = dir_output+outputfile
    write(output,track)
