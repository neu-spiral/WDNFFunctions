from Toplogy_gen  import Problem, Demand
import time
import argparse
from cvxopt import matrix
from Continuous_Greedy import write
from random_replacement import OBJ_dict, ro_uvr
from poly import rho_uv_dicts, poly
import os
import numpy as np

def Dependencies(P):
    """Find which edges an element (v,i) affects, in general."""
    dep = {}

    for demand in P.demands:
        path = demand.path
        item = demand.item
        path_len = len(path)
        if path_len > 1:
            v = path[path_len-2]
            depend = [(path[path_len-2],path[path_len-1])]
            if (v,item) in dep:
                dep[(v,item)][demand] = depend
            else:
                dep[(v,item)] = {demand: depend}
            for i in range(path_len-3,-1,-1):
                v = path[i]
                u = path[i+1]
                depend = [(v,u)] + dep[(u,item)][demand]
                if (v,item) in dep:
                    dep[(v,item)][demand]= depend
                else:
                    dep[(v,item)] = {demand: depend}
        else:
            pass
    return dep

def RhoPoly(P):
    rho_uvs_coefficients,  rho_uvs_sets = rho_uv_dicts(P.demands, P.EDGE)
    rho_poly = {}
    for edge in rho_uvs_coefficients:
        rho_poly[edge]  = {}
        for demand in rho_uvs_coefficients[edge]:
            rho_poly[edge][demand] = poly(rho_uvs_coefficients[edge][demand], rho_uvs_sets[edge][demand]) # construct rho_e class
    return rho_poly

def problem_size(P):
    dem_items = []
    for demand in P.demands:
        item = demand.item
        dem_items.append(item)
    V = len(P.capacities)
    I = len(dem_items)
    return V,I


def GREEDY(P):
    V,I = problem_size(P)
    track = []
    X = np.zeros([V,I])
    free_capacities = dict(P.capacities)
    cardinality = sum(P.capacities.values())

    tstart = time.time()
    ro_dict = ro_uvr(P, X)

    OBJ0 = OBJ_dict(P, ro_dict)
    track.append(OBJ0)

    print "OBJ 0 is", OBJ0

    '''uniformly assign service rate'''
    for edge in P.EDGE:
        num_demand = len(P.EDGE[edge])
        new_mu = P.capacity_servicerate[edge] / num_demand
        for demand in P.EDGE[edge]:
            P.EDGE[edge][demand] = new_mu

    '''calculate the new objective with uniform service rate'''
    ro_dict = ro_uvr(P, X)
    OBJ = OBJ_dict(P, ro_dict)

    elapsed = time.time() - tstart
    track.append((elapsed, OBJ0 - OBJ))
    rho_poly = RhoPoly(P)
    dependencies = Dependencies(P)
    while(cardinality>0):

        max_delta = 0.
        for (v,i) in dependencies:

            if free_capacities[v]>0:
                delta_vi = 0.
                '''calculate the decrease by placing item i on node v'''
                for demand in dependencies[(v,i)]:
                    for edge in dependencies[(v, i)][demand]:
                        r_polys = rho_poly[edge][demand]
                        r = r_polys.evaluate(X)

                        delta = P.utilityfunction(r,0)
                        delta_vi += delta

                ''' Record the maximum decrease'''
                if delta_vi > max_delta:
                    new_element = (v,i)
                    max_delta = delta_vi
                else:
                    pass
            else:
                pass
        '''there is no use to place any item'''
        if max_delta<=0.:
            break
        else:

            OBJ -= max_delta
            v_new,i_new = new_element
            '''in case place same item to the node twice'''
            del dependencies[new_element]

            X[new_element] = 1

            free_capacities[v_new] -=1
            cardinality -= 1
            elapsed = time.time() - tstart
            track.append((elapsed, OBJ0 - OBJ))

#    ro_dict = ro_uvr(P, X)
#    OBJ = OBJ_dict(P, ro_dict)
    track.append(X)
    return X,track

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Simulate the Greedy Alg.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--order_moment', default=2, type=int, help='Order of moment for the expected cost', choices=[1,2,3,4])
    parser.add_argument('--queue_type', default='MMInfty', type=str, help='Type of queue', choices=['MMInfty', 'Info'])
    args  = parser.parse_args()
    problem_instance = "problem_" + args.graph_type + "_1000demands_100catalog_size_2mincap_2maxcap_100size_powerlaw_rate1.0_" + str(
        args.query_nodes) + "qnodes_" + str(args.order_moment) + "order_" + args.queue_type
    input = "INPUT/"+problem_instance
    
    P = Problem.unpickle_cls(input)


    X, track = GREEDY(P)
    print track[-2]
    dir_output = "GREEDY/"
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    outputfile = problem_instance
    output = dir_output+outputfile
    np.save(output,track)
