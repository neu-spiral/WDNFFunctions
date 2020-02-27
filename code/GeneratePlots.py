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
    plt.figure()
    # greedy_track = load("results/greedy/IM_epinions100_recall")
    # sys.stderr.write("greedy track is: " + str(greedy_track))
    # utility = [item[1][1] for item in greedy_track.items()]
    # cardinality = [item[0] for item in greedy_track.items()]
    # plt.plot(cardinality, utility, label='Greedy Algorithm')
    # plt.xlabel('Cardinality')
    # plt.ylabel('Utility')
    # plt.legend()
    # plt.savefig('results/plots/GreedyAlgorithmEpinions100IM3.png')
    # cont_greedy_track_samp = load("results/continuous_greedy/IM_Epinions100_samplerWithDependencies_30_FW_30samples")
    # sys.stderr.write("cont_greedy track is: " + str(cont_greedy_track))
    # utility_swd = [item[2] for item in cont_greedy_track_samp]
    # cardinality_swd = [item[0] for item in cont_greedy_track_samp]
    # cont_greedy_track_poly = load("results/continuous_greedy/IM_Epinions100_polynomial_30_FW_degree_2_around_0.0")
    # utility_p = [item[2] for item in cont_greedy_track_poly]
    # cardinality_p = [item[0] for item in cont_greedy_track_poly]
    # plt.plot(cardinality, utility, "b--", cardinality_swd, utility_swd, "g^", cardinality_p, utility_p, "ro")
    # plt.xlabel('Cardinality')
    # plt.ylabel('Utility')
    # plt.legend()
    # plt.savefig('results/plots/Comparisons_on_test_graphs.png')
    # for file in os.listdir("results/continuous_greedy"):
    #    if "IMtest_casepolynomial_100_FW_degree_4_around_0.0" in file or 'IMtest_casesampler_100_FW_100samples' in file:

         # sys.stderr.write("cont_greedy track is: " + str(cont_greedy_track))
    #        utility = []
    #        cardinality = []
    #        for item in cont_greedy_track:
    #            utility.append(item[2])
    #            cardinality.append(item[0])
    #        plt.plot(cardinality, utility, "^", label=file)
    # plt.legend(fontsize='xx-small')
    # plt.savefig('results/plots/sample_vs_poly_on_test_graphs.png')

    utility1 = []
    samples1 = []
    cont_greedy_track1 = load("results/continuous_greedy/IMtest_case_sampler_estimation")
    for item in cont_greedy_track1:
        utility1.append(item[2])
        samples1.append(item[1])
    utility2 = []
    samples2 = []
    cont_greedy_track2 = load("results/continuous_greedy/IMtest_case_sampler_with_dep_estimation")
    for item in cont_greedy_track2:
        utility2.append(item[2])
        samples2.append(item[1])
    plt.plot(samples1, utility1, "^", samples2, utility2, "ro")
    # plt.legend(fontsize='xx-small')
    plt.xlabel('Number of Samples')
    plt.ylabel('Utility')
    plt.savefig('results/plots/numOfSamplesVSUtilityOnSamplerAndDepSampler.png')

    # sampler_obj = eval(open('sampler_obj.txt', 'r').read())
    # iterations1 = list(range(1, len(sampler_obj) + 1))

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
