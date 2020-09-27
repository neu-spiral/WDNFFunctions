import random
from random_replacement import ro_uvr,OBJ_dict
from cvxopt import matrix
import time
import argparse
from pyspark import SparkContext, SparkConf
from helpers import write
from Toplogy_gen import Problem, Demand
import os
import numpy as np

def SwapRound(beta_bases_tpl,v):
     
    def MergeBases(beta1, B1, beta2, B2,random_generator):
        def subtract(B1,B2):
            """Return B1-B2"""
            a= set(B1)
            b=set(B2)
            return tuple(a.difference(b))
        k=0
        while set(B1)!=set(B2):
            B1_slsh_B2 = subtract(B1,B2)
            B2_slsh_B1 = subtract(B2,B1)
            i = B1_slsh_B2[0]
            j = B2_slsh_B1[0]
            RN = random_generator.random()
            if RN<beta1/(beta1+beta2):
                B2 = subtract(B2,(j,))+(i,)
            else:
                B1 = subtract(B1,(i,))+(j,)
            k+=1
        return B1
    RG = random.Random()
    RG.seed(v)
    p = len(beta_bases_tpl)
    C_new = beta_bases_tpl[0][0]
    beta1 = beta_bases_tpl[0][1]
    for k in range(p-1):
        beta2 = beta_bases_tpl[k+1][1]
        B2 = beta_bases_tpl[k+1][0]
        C_old = C_new
         
        C_new = MergeBases(beta1=beta1,B1=C_old,beta2=beta2,B2=B2,random_generator=RG)
        beta1 = beta1 +beta2
    return C_new


def distributed_rounding(input,P):
    def flatten_dictionary(dict):
        l = []
        for v in dict:
           l.append((v,dict[v]))
        return l
    def make_unique(l):
        l.sort()
        return tuple(l)
    f = open(input, 'r')
    base_list= eval(f.readline())
    f.close()
    conf = (SparkConf()
         .setMaster("local[40]")
         .setAppName("My app")
         .set("spark.executor.memory", "100g"))
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    RDD_bases = sc.parallelize(base_list).flatMap(flatten_dictionary).mapValues(lambda base_list:make_unique(base_list))\
                                         .map(lambda (v,tpl_base):((v,tpl_base),1)).reduceByKey(lambda x,y:x+y, P)\
                                         .map(lambda ((v,base_tpl),cnt):(v,[(base_tpl,cnt)])).reduceByKey(lambda x,y:x+y, P)\
                                         .map(lambda (v,base_tpls):(v,(base_tpls,v))).partitionBy(P)\
             			         .mapValues(lambda (base_tpls,v):SwapRound(base_tpls,v)).cache()
                                         
    return RDD_bases 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Simulate the Swap Rounding Alg.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('OBJ', type=str, help='Objective function queuing type', choices=['MMInfty', 'Info'])
    parser.add_argument('SOLUTION', type=str, help='Solution queuing type', choices=['MMInfty', 'Info'])
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--query_nodes', default=10, type=int, help='Number of nodes generating queries')

    parser.add_argument('--order_moment', default=2, type=int, help='Order of moment for the expected cost', choices=[1,2,3,4])
    parser.add_argument('--estimator',default='sample',choices=['sample','taylor','power','DR'],help='Type of estimator')
    parser.add_argument('--k',type=int, default=1,help="Order of taylor expansion")
    args  = parser.parse_args()
    problem_instance = "problem_" + args.graph_type + "_1000demands_100catalog_size_2mincap_2maxcap_100size_powerlaw_rate1.0_" + str(args.query_nodes) + "qnodes_" + str(args.order_moment) + "order_" + args.SOLUTION
    if args.estimator == "sample":
        input = problem_instance + "-500-samples-1000-iters"
    elif args.estimator == "taylor":
        input = problem_instance + "-taylor-k-" + str(args.k)+ "-1000-iters"
    else:
        input = problem_instance
    OBJ = "problem_" + args.graph_type + "_1000demands_100catalog_size_2mincap_2maxcap_100size_powerlaw_rate1.0_" + str(args.query_nodes) + "qnodes_" + str(args.order_moment) + "order_" + args.OBJ

    input = "BASE/"+ input
    input_problem = "INPUT_NEW/"+problem_instance
    obj = "INPUT/" + OBJ
    P = Problem.unpickle_cls(input_problem)
    P_obj = Problem.unpickle_cls(obj)
    capacities = P.capacities
    dem_items=[]
    for demand in P.demands:
         item = demand.item
         dem_items.append(item)
    V,I = (len(P.capacities),len(set(dem_items)))
    RDD_bases = distributed_rounding(input,56)
    base_out  = RDD_bases.collect()
    X = np.zeros([V,I])
    tstart = time.time()
    for (v,items) in base_out:
         for item in items:
             X[v,item] = 1
    print X
    ro_uvr_dic = ro_uvr(P,X)
    OBJ =  OBJ_dict(P_obj,ro_uvr_dic)
    elapsed = time.time()-tstart
    print "OBJ is %f and time taken is %f" %(P.cost-OBJ,elapsed)
    track = [P.cost]
    track += [(elapsed,P.cost-OBJ)]
    track += [X]

    dir_rounded = "ROUNDED/"
    if not os.path.exists(dir_rounded):
        os.mkdir(dir_rounded)

    if args.OBJ == args.SOLUTION:
        output = input
    else:
        if args.estimator == "sample":
            output = problem_instance + "2-500-samples-1000-iters"
        elif args.estimotor == "taylor":
            output = problem_instance + "2-taylor-k-" + str(args.k) + "-1000-iters"
        else:
            output = problem_instance + "2"

    rounded = dir_rounded + output
    np.save(rounded, track)
