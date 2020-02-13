import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from matplotlib.transforms import Bbox
import numpy as np
from matplotlib.dates import date2num
import datetime
import os
import sys
from helpers import load



if __name__ == "__main__":

    #time_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_time", "r").read())
    #utility_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_utilities", "r").read())
    greedy_track = load("results/greedy/IM_epinions100_recalc_")
    sys.stderr.write("greedy track is: " + str(greedy_track))
    utility = [item[1][1] for item in greedy_track.items()]
    cardinality = [item[0] for item in greedy_track.items()]
    plt.plot(cardinality, utility, label = 'Greedy Algorithm')
    plt.xlabel('Cardinality')
    plt.ylabel('Utility')
    plt.legend()
    plt.savefig('results/plots/GreedyAlgorithmEpinions100IM2.png')
    # cont_greedy_track = load("results/continuous_greedy/IM_Epinions100_samplerWithDependencies_100_FW_500samples")
    # sys.stderr.write("cont_greedy track is: " + str(cont_greedy_track))
    # utility_swd = [item[2] for item in cont_greedy_track]
    # cardinality_swd = [item[0] for item in cont_greedy_track]
    # plt.plot(cardinality, utility, "b--", cardinality_swd, utility_swd, "g^")
    # plt.xlabel('Cardinality')
    # plt.ylabel('Utility')
    # plt.legend()
    # plt.savefig('results/plots/Comparisons_on_Epinions100IM.png')
    #sampler_obj = eval(open('sampler_obj.txt', 'r').read())
    #iterations1 = list(range(1, len(sampler_obj) + 1))

    #poly_obj = eval(open('poly_obj.txt', 'r').read())
    #iterations2 = list(range(1, len(poly_obj) + 1))

    #plt.plot(time_ax, utility_ax, label = '2nd degree Polynomial Estimator around 0.0')
    #plt.plot(iterations1, sampler_obj, 'g^', label = 'Sampler Estimator')
    #plt.xlabel('Time')
    #plt.ylabel('Utility')
    #plt.legend()
    #plt.savefig('results/plots/2ndDegreePolynomialEstimatorAround00.png')

    #sampler_time = np.log(eval(open('sampler_time.txt', 'r').read()))
    #poly_time = np.log(eval(open('poly_time.txt', 'r').read()))

    #fig, ax = plt.subplots()
    #bar_width = 0.35
    #opacity = 0.8
    #iterations2 = [x + bar_width for x in iterations1]
    #rects1 = plt.bar(iterations1, sampler_time, bar_width, alpha = opacity, color = 'g', label = 'Sampler Estimator')
    #rects2 = plt.bar(iterations2, poly_time, bar_width, alpha = opacity, color = 'b', label = 'Polynomial Estimator')
    #plt.xlabel('Number of Iterations')
    #plt.ylabel('Time')
    #plt.xticks(, iterations)
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig('IterVSTime.png')
