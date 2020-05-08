from ContinuousGreedy import multilinear_relaxation
from helpers import load
import argparse
import datetime
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter for results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default='results/continuous_greedy/IM/epinions_100_10cascades/k_10_100_FW', type=str,
                        help='Input file for the plots')
    parser.add_argument('--type', default='SEEDSvsUTILITY', type=str, help='Type of the plot',
                        choices=['TIMEvsUTILITY', 'LOGTIMEvsUTILITY', 'ITERATIONSvsUTILITY', 'SEEDSvsUTILITY'])
    args = parser.parse_args()

    path = args.input  # "results/continuous_greedy/FL/ratings_10/k_3_100_FW"
    # "results/continuous_greedy/IM/random10/k_4_100_FW"
    files = os.listdir(path)
    if args.type == 'TIMEvsUTILITY':
        plt.figure()
        for file in files:
            result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
            # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
            # args.samples)
            # track = result[0]
            # utility_function = result[1]
            solutions = []  # fractional vectors y
            objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
            time = []  # time it took to compute the fractional vector y
            FW_iterations = []
            degree = []  # degree of the polynomial estimator
            center = []  # point where the polynomial estimator is centered
            samples = []  # number of samples used in the sampler estimator
            for item in result:
                FW_iterations.append(item[0])
                time.append(item[1])
                solutions.append(item[2])
                objectives.append(item[3])
            my_label = file
            plt.plot(time, objectives, 's', label=my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Time (seconds)")
        plt.ylabel("f^(y)")
        plt.legend(fontsize='xx-small')
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_time.png', bbox_inches="tight")

    elif args.type == 'LOGTIMEvsUTILITY':
        plt.figure()
        for file in files:
            result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
            # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
            # args.samples)
            # track = result[0]
            # utility_function = result[1]
            solutions = []  # fractional vectors y
            objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
            time = []  # time it took to compute the fractional vector y
            FW_iterations = []
            degree = []  # degree of the polynomial estimator
            center = []  # point where the polynomial estimator is centered
            samples = []  # number of samples used in the sampler estimator
            for item in result:
                FW_iterations.append(item[0])
                time.append(item[1])
                # time.append(np.log(item[1]))
                solutions.append(item[2])
                objectives.append(item[3])
            my_label = file
            # plt.plot(FW_iterations, objectives, 's', label=my_label)
            plt.semilogx(time, objectives, 's', label=my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Log Time (seconds)")
        plt.ylabel("f^(y)")
        plt.legend(fontsize='xx-small')
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_logtime.png', bbox_inches="tight")

    elif args.type == 'ITERATIONSvsUTILITY':
        plt.figure()
        for file in files:
            result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
            # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
            # args.samples)
            # track = result[0]
            # utility_function = result[1]
            solutions = []  # fractional vectors y
            objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
            time = []  # time it took to compute the fractional vector y
            FW_iterations = []
            degree = []  # degree of the polynomial estimator
            center = []  # point where the polynomial estimator is centered
            samples = []  # number of samples used in the sampler estimator
            for item in result:
                FW_iterations.append(item[0])
                solutions.append(item[2])
                objectives.append(item[3])
            my_label = file
            plt.plot(FW_iterations, objectives, 's', label=my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Iterations")
        plt.ylabel("f^(y)")
        plt.legend(fontsize='xx-small')
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '.png', bbox_inches="tight")

    elif args.type == 'SEEDSvsUTILITY':
        seeds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        utility1 = []
        utility2 = []
        for seed in seeds:
            path1 = 'results/continuous_greedy/IM/epinions_100_10cascades/k_' + str(seed) + '_100_FW/polynomial_degree_1_around_05'
            path2 = 'results/continuous_greedy/IM/epinions_100_10cascades/k_' + str(
                seed) + '_100_FW/polynomial_degree_2_around_05'
            result1 = load(path1)
            result2 = load(path2)
            utility1.append(result1[-1][3])
            utility2.append(result2[-1][3])
        plt.figure()
        plt.plot(seeds, utility1, 's', label='Polynomial Estimator degree 1')
        plt.plot(seeds, utility2, 's', label='Polynomial Estimator degree 2')
        plt.title("Number of seeds vs utility")
        plt.xlabel("Constraints")
        plt.ylabel("f^(y)")
        plt.legend(fontsize='xx-small')
        plt.show()
        plt.savefig('seeds.png', bbox_inches="tight")

        # plt.figure()
        # for file in files:
        #     result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
        #     # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
        #     # args.samples)
        #     # track = result[0]
        #     # utility_function = result[1]
        #     solutions = []  # fractional vectors y
        #     objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
        #     time = []  # time it took to compute the fractional vector y
        #     FW_iterations = []
        #     degree = []  # degree of the polynomial estimator
        #     center = []  # point where the polynomial estimator is centered
        #     samples = []  # number of samples used in the sampler estimator
        #     for item in result:
        #         FW_iterations.append(item[0])
        #         # time.append(item[1])
        #         time.append(np.log(item[1]))
        #         solutions.append(item[2])
        #         objectives.append(item[3])
        #     my_label = file
        #     # plt.plot(FW_iterations, objectives, 's', label=my_label)
        #     plt.plot(time, objectives, 's', label=my_label)
        # title_str = path.split("/")[-1].split("_")
        # plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        # # plt.xlabel("Iterations")
        # plt.xlabel("Log Time")
        # # plt.xlabel("Time")
        # plt.ylabel("Multilinear Relaxation")
        # plt.legend(fontsize='xx-small')
        # plt.show()
        # output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # # plt.savefig(output_dir + '_time.png', bbox_inches="tight")
        # # plt.savefig(output_dir + '.png', bbox_inches="tight")
        # plt.savefig(output_dir + '_logtime.png', bbox_inches="tight")

    #time_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_time", "r").read())
    #utility_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_utilities", "r").read())
    # plt.figure()
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

    # utility1 = []
    # samples1 = []
    # time1 = []
    # degrees1 = []
    # out = []
    # cont_greedy_track1 = load("results/continuous_greedy/IM_random10_sampler_y_random")
    # for item in cont_greedy_track1:
    #     #if item[1] < 21:
    #     utility1.append(item[2])
    #     # degrees1.append(item[1])
    #     #samples1.append(item[1])
    #     time1.append(np.log(item[0]))
    #
    # utility2 = []
    # samples2 = []
    # time2 = []
    # degrees2 = []
    # cont_greedy_track2 = load("results/continuous_greedy/IM_random10_polynomial_0.5_y_random")
    # for item in cont_greedy_track2:
    #     #if item[1] < 21:
    #     utility2.append(item[2])
    #     #degrees2.append(item[1])
    #         #samples2.append(item[1])
    #     time2.append(np.log(item[0]))
    #     out.append(item[3])
    #     #iterations2.append(item[0])
    #     #mlr2.append(item[1])
    # # print(out)
    # # plt.plot(time1, utility1, "^", label="Sampler")
    # # plt.plot(iterations2, mlr2, "ro", label="Sampler with Dependencies")
    # # plt.legend(fontsize='xx-small')
    # # plt.xlabel('Iterations')
    # # plt.ylabel('Multilinear Relaxation')
    # # plt.savefig('results/plots/multivsiterations.png')
    #
    # # plt.figure()
    # # utility3 = []
    # # degrees = []
    # # time3 = []
    # # cont_greedy_track3 = load("results/continuous_greedy/IM_1_graph_y_rand2_poly0.0")
    # # for item in cont_greedy_track3:
    # #    utility3.append(item[2])
    # #    degrees.append(item[1])
    # #    time3.append(np.log(item[0]))
    # #    print(item)
    # # plt.plot(degrees, utility3, "^", label="Polynomial Estimation")
    # # plt.legend()
    # # plt.xlabel('Degree of the expansion')
    # # plt.ylabel('Utility')
    # # plt.savefig('results/plots/1Graph_y_rand2_poly0.5_DegreeVSUtility.png')
    #
    # plt.figure()
    # plt.plot(time1, utility1, "g^", label="Sampler", markersize=5)
    # # plt.plot(time2, utility2, "ro", label="Sampler with Dependencies", markersize=5)
    # plt.plot(time2, utility2, "bx", label="Polynomial", markersize=5)
    # plt.plot(time2, out, label="Multilinear Relaxation")
    # plt.legend(fontsize='xx-small')
    # plt.xlabel('Time')
    # plt.ylabel('Utility')
    # plt.savefig('results/plots/Random10_y_random_LogTimeVSUtilityOnSamplerAndDepSamplerAndPolynomial05.png')

    ## UNCOMMENT HERE FOR COMPARING POLYNOMIAL ESTIMATORS CENTERED AROUND DIFFERENT POINTS #
    # plt.figure()
    # plt.plot(degrees1, utility1, "g^", label="Polynomial around 0.0", markersize=5)
    # plt.plot(degrees2, utility2, "ro", label="Polynomial around 0.5", markersize=5)
    # # plt.plot(time3, utility3, "bx", label="Polynomial", markersize=5)
    # plt.plot(degrees2, out, label="Multilinear Relaxation")
    # plt.legend(fontsize='xx-small')
    # plt.xlabel('Degree')
    # plt.ylabel('Utility')
    # plt.title('Comparison of polynomial estimators around 0.0 and 0.5')
    # plt.savefig('results/plots/1Graph_y_05_DegreeVSUtilityOnPolynomials00and05.png')
    ##STOP UNCOMMENTING HERE #

    # # UNCOMMENT HERE FOR COMPARING MLR RESULTS OF ESTIMATORS #
    # iterations1 = []
    # mlr1 = []
    # cont_greedy_track1 = load("results/continuous_greedy/DR_epinions_20_polynomial_50_FW_degree_4_around_0.5")
    # for item in cont_greedy_track1:
    #     iterations1.append(item[0])
    #     mlr1.append(item[1])
    #
    # iterations2 = []
    # mlr2 = []
    # cont_greedy_track2 = load("results/continuous_greedy/DR_epinions_20_sampler_50_FW_500_samples")
    # for item in cont_greedy_track2:
    #     iterations2.append(item[0])
    #     mlr2.append(item[1])
    #
    # x_lims = [0, 50]
    # x1 = np.linspace(x_lims[0], x_lims[1], 1000)
    #
    # plt.plot(iterations1, mlr1, "^", label="Polynomial Estimator")
    # plt.plot(iterations2, mlr2, "ro", label="Sampler Estimator")
    # # plt.plot(x1, np.log1p(x1/50.0), label='log(x + 1)')
    # plt.legend(fontsize='xx-small')
    # plt.xlabel('Iterations')
    # plt.ylabel('Multilinear Relaxation')
    # plt.savefig('results/plots/multivsiterationsDR0.png')
    # #STOP UNCOMMENTING HERE #

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
