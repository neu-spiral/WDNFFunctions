from helpers import load, save
from time import time
#from Toplogy_gen  import Problem, Demand
from ProblemInstances import InfluenceMaximization
from networkx import Graph, DiGraph
import argparse
#from cvxopt import matrix
#from Continuous_Greedy import write
#from random_replacement import OBJ_dict, ro_uvr
#from poly import rho_uv_dicts, poly
import logging
import numpy as np
import os
import sys

#def Dependencies(P):
#    """Find which edges an element (v,i) affects, in general."""
#    dep = {}
#
#    for demand in P.demands:
#        path = demand.path
#        item = demand.item
#        path_len = len(path)
#        if path_len > 1:
#            v = path[path_len-2]
#            depend = [(path[path_len-2],path[path_len-1])]
#            if (v,item) in dep:
#                dep[(v,item)][demand] = depend
#            else:
#                dep[(v,item)] = {demand: depend}
#            for i in range(path_len-3,-1,-1):
#                v = path[i]
#                u = path[i+1]
#                depend = [(v,u)] + dep[(u,item)][demand]
#                if (v,item) in dep:
#                    dep[(v,item)][demand]= depend
#                else:
#                    dep[(v,item)] = {demand: depend}
#        else:
#            pass
#    return dep

#def RhoPoly(P):
#    rho_uvs_coefficients,  rho_uvs_sets = rho_uv_dicts(P.demands, P.EDGE)
#    rho_poly = {}
#    for edge in rho_uvs_coefficients:
#        rho_poly[edge]  = {}
#        for demand in rho_uvs_coefficients[edge]:
#            rho_poly[edge][demand] = poly(rho_uvs_coefficients[edge][demand], rho_uvs_sets[edge][demand]) # construct rho_e class
#    return rho_poly

#def problem_size(P):
#    dem_items = []
#    for demand in P.demands:
#        item = demand.item
#        dem_items.append(item)
#    V = len(P.capacities)
#    I = len(dem_items)
#    return V,I


#def GREEDY(P):
#    V,I = problem_size(P)
#    track = []
#    X = np.zeros([V,I])
#    free_capacities = dict(P.capacities)
#    cardinality = sum(P.capacities.values())

#    tstart = time.time()
#    ro_dict = ro_uvr(P, X)

#    OBJ0 = OBJ_dict(P, ro_dict)
#    track.append(OBJ0)

#    print "OBJ 0 is", OBJ0

#    '''uniformly assign service rate'''
#    for edge in P.EDGE:
#        num_demand = len(P.EDGE[edge])
#        new_mu = P.capacity_servicerate[edge] / num_demand
#        for demand in P.EDGE[edge]:
#            P.EDGE[edge][demand] = new_mu

#    '''calculate the new objective with uniform service rate'''
#    ro_dict = ro_uvr(P, X)
#    OBJ = OBJ_dict(P, ro_dict)

#    elapsed = time.time() - tstart
#    track.append((elapsed, OBJ0 - OBJ))
#    rho_poly = RhoPoly(P)
#    dependencies = Dependencies(P)
#    while(cardinality>0):

#        max_delta = 0.
#        for (v,i) in dependencies:

#            if free_capacities[v]>0:
#                delta_vi = 0.
#                '''calculate the decrease by placing item i on node v'''
#                for demand in dependencies[(v,i)]:
#                    for edge in dependencies[(v, i)][demand]:
#                        r_polys = rho_poly[edge][demand]
#                        r = r_polys.evaluate(X)

#                        delta = P.utilityfunction(r,0)
#                        delta_vi += delta

#                ''' Record the maximum decrease'''
#                if delta_vi > max_delta:
#                    new_element = (v,i)
#                    max_delta = delta_vi
#                else:
#                    pass
#            else:
#                pass
#        '''there is no use to place any item'''
#        if max_delta<=0.:
#            break
#        else:

#            OBJ -= max_delta
#            v_new,i_new = new_element
#            '''in case place same item to the node twice'''
#            del dependencies[new_element]

#            X[new_element] = 1

#            free_capacities[v_new] -=1
#            cardinality -= 1
#            elapsed = time.time() - tstart
#            track.append((elapsed, OBJ0 - OBJ))

#    ro_dict = ro_uvr(P, X)
#    OBJ = OBJ_dict(P, ro_dict)
#    track.append(X)
#    return X,track


def greedy(problem):
    y = dict.fromkeys(problem.groundSet, 0.0)
    # sys.stderr.write("y is: " + str(y))
    unchosen_elements = dict(filter(lambda element: element[1] == 0.0, y.items()))
    # sys.stderr.write("unchosen_elements are: " + str(unchosen_elements))
    objective0 = problem.utility_function(y)
    objective = objective0
    # sys.stderr.write("objective is: " + str(objective0) + "\n")
    cardinality = problem.problemSize
    track = dict()
    k = cardinality
    start = time()
    track[0] = (start, objective0)
    dependencies = problem.dependencies.keys()
    while cardinality > 0:
        max_delta = 0.0
        objective0 = problem.utility_function(y)
        for element in unchosen_elements:
            if element in dependencies:
                y[element] = 1
                # sys.stderr.write("y is: " + str(y))
                objective = problem.utility_function(y)
                # sys.stderr.write("objective is: " + str(objective) + "\n")
                y[element] = 0
                delta = objective - objective0
                if delta > max_delta:
                    selection = element
                    max_delta = delta
        if max_delta > 0.0:
            del unchosen_elements[selection]
            logging.info('...' + str(selection) + ' is just selected...' + "\n")
            y[selection] = 1
            objective += max_delta
            cardinality -= 1
            elapsed_time = time() - start
            track[k - cardinality] = (elapsed_time, objective)
        else:
            break
    return y, track









if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Simulate the Greedy Algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--problem_type', default = 'IM', type = str, help = 'Type of the problem', choices = ['DR', 'QS', 'IM', 'FL'])
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'epinions100'])
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')
    parser.add_argument('--order_moment', default=2, type=int, help='Order of moment for the expected cost', choices=[1,2,3,4])
    parser.add_argument('--queue_type', default='MMInfty', type=str, help='Type of queue', choices=['MMInfty', 'Info'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    #problem_instance = "problem_" + args.graph_type + "_1000demands_100catalog_size_2mincap_2maxcap_100size_powerlaw_rate1.0_" + str(
    #    args.query_nodes) + "qnodes_" + str(args.order_moment) + "order_" + args.queue_type
    #input = "INPUT/"+problem_instance
    
    #P = Problem.unpickle_cls(input)


    #X, track = GREEDY(P)
    #print track[-2]
    logging.info('Greedy Algorithm is initiated...')
    directory_output = "results/greedy/"
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
    logging.info('...output directory is created...')
    #outputfile = problem_instance
    #output = dir_output+outputfile
    #np.save(output,track)

    if args.problem_type == "IM":
        logging.info('Problem Type is selected as Influence Maximization...')
        problem_instance = "IM_" + "epinions100_recall"
        logging.info('Loading graphs list...')
        graphs = load("datasets/test_graphs_file")
        # new_graph = DiGraph()
        # new_graph.add_nodes_from([1, 2, 3, 4, 5, 6])
        # new_graph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5), (4, 6), (6, 3)])
        # graphs = [new_graph]
        # sys.stderr.write("graphs is: " + str(graphs))
        logging.info('...done. Initiating the Influence Maximization Problem...')
        newProblem = InfluenceMaximization(graphs, 3)
        logging.info('...done. Starting the greedy algorithm...')
        y, track = greedy(newProblem)
        logging.info('...done.')
        output = directory_output + problem_instance
        logging.info('Saving the results of the greedy algorithm...')
        save(output, track)
        logging.info('...done. Simulation is finished.')
