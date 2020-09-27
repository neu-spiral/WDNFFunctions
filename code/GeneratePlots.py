from ContinuousGreedy import multilinear_relaxation
from helpers import load
from ProblemInstances import DiversityReward, QueueSize, InfluenceMaximization, FacilityLocation, derive
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
# from pylab import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter for results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default='results/continuous_greedy/IM/epinions_100_10cascades/k_10_100_FW', type=str,
                        help='Input file for the plots')
    parser.add_argument('--type', default='SEEDSvsUTILITY', type=str, help='Type of the plot',
                        choices=['TIMEvsUTILITY', 'LOGTIMEvsUTILITY', 'ITERATIONSvsUTILITY', 'SEEDSvsUTILITY',
                                 'PARETO', 'PARETOLOG', 'CONV_TEST'])
    parser.add_argument('--font_name', type=str, help='Font of the title and axes')
    parser.add_argument('--font_size', default=14, type=int, help='Font size of the title and axes')
    args = parser.parse_args()
    n = 10  # plot every nth element from a list
    font_name = args.font_name
    font_size = args.font_size
    plt.rcParams.update({'font.size': font_size})

    path = args.input  # "results/continuous_greedy/FL/ratings_10/k_3_100_FW"
    # "results/continuous_greedy/IM/random10/k_4_100_FW"/
    k = int(path.split('/')[-1].split('_')[1])
    files = os.listdir(path)
    if args.type == 'TIMEvsUTILITY':
        plt.figure()
        for file in files:
            if "backup" in file:
                problem_file = path.replace('/', '_')\
                                   .replace('_100_FW', '')\
                                   .replace('results_continuous_greedy_', 'problems/')
                problem = load(problem_file)
                result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                track = result[1]
                time = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                for key in track:
                    # FW_iterations.append(key)
                    time.append(track[key][0])
                    objectives.append(problem.utility_function(track[key][1]))
                my_label = file
                if 'polynomial' in file:
                    my_marker = "o"
                elif 'samplerWith' in file:
                    my_marker = "^"
                else:
                    my_marker = "*"
                plt.plot(time[0::n], objectives[0::n], marker=my_marker, label=my_label)

            else:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form
                # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                # track[key][1]), args.estimator, args.samples)
                time = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                solutions = []  # fractional vectors y
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                degree = []  # degree of the polynomial estimator
                center = []  # point where the polynomial estimator is centered
                samples = []  # number of samples used in the sampler estimator
                for item in result:
                    # FW_iterations.append(item[0])
                    time.append(datetime.timedelta(seconds=item[1]))  # datetime.datetime.(item[1])
                    # solutions.append(item[2])
                    objectives.append(item[3])
                my_label = file
                if 'polynomial' in file:
                    my_marker = "o"
                elif 'samplerWith' in file:
                    my_marker = "^"
                else:
                    my_marker = "*"
                plt.plot(time[0::n], objectives[0::n], marker=my_marker, label=my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Time (seconds)")
        plt.ylabel("f^(y)")
        plt.legend()  # fontsize='xx-small')
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_time.pdf', bbox_inches="tight")

    elif args.type == 'LOGTIMEvsUTILITY':
        plt.figure()
        for file in files:
            if "backup" in file:
                problem_file = path.replace('/', '_')\
                                   .replace('_100_FW', '')\
                                   .replace('results_continuous_greedy_', 'problems/')
                problem = load(problem_file)
                result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                track = result[1]
                time = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                for key in track:
                    # FW_iterations.append(key)
                    time.append(track[key][0])
                    objectives.append(problem.utility_function(track[key][1]))
                my_label = file
                if 'polynomial' in file:
                    my_marker = "o"
                elif 'samplerWith' in file:
                    my_marker = "^"
                else:
                    my_marker = "*"
                plt.semilogx(time[0::n], objectives[0::n], marker=my_marker, label=my_label)

            else:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
                # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
                # args.samples)
                # track = result[0]
                # utility_function = result[1]
                # solutions = []  # fractional vectors y
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                time = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                degree = []  # degree of the polynomial estimator
                center = []  # point where the polynomial estimator is centered
                samples = []  # number of samples used in the sampler estimator
                for item in result:
                    # FW_iterations.append(item[0])
                    time.append(item[1])
                    # time.append(np.log(item[1]))
                    # solutions.append(item[2])
                    objectives.append(item[3])
                    number = item[5]
                if 'polynomial' in file:
                    my_marker = "x"
                    my_label = 'POLY' + str(number)
                    my_line = 'dashed'
                elif 'samplerWith' in file:
                    my_marker = "o"
                    my_label = 'SWD' + str(number)
                    my_line = 'dashdot'
                else:
                    my_marker = "^"
                    my_label = 'SAMP' + str(number)
                    my_line = 'dashdot'
                if number > 100 or number < 5:
                    plt.semilogx(time[0::n], objectives[0::n], marker=my_marker, label=my_label, linestyle=my_line)
                    print(time[99], objectives[99], my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Log Time (seconds)")
        plt.ylabel(r'$f (\mathbf{y})$')  #, fontsize=12)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        # labels, handles = zip(sorted(zip(labels, handles), key=lambda t: t[0]))
        labels.sort()
        ax.legend(handles, labels)  # fontsize='medium')
        plt.legend(fontsize='small', bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_logtime.pdf', bbox_inches="tight")
        plt.savefig(output_dir + '_logtime.png', bbox_inches="tight")

    elif args.type == 'ITERATIONSvsUTILITY':
        plt.figure()
        for file in files:
            if "backup" in file:
                problem_file = path.replace('/', '_')\
                                   .replace('_100_FW', '')\
                                   .replace('results_continuous_greedy_', 'problems/')
                problem = load(problem_file)
                result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                track = result[1]
                # time = []  # time it took to compute the fractional vector y
                FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                for key in track:
                    FW_iterations.append(key)
                    # time.append(track[key][0])
                    objectives.append(problem.utility_function(track[key][1]))
                my_label = file
                if 'polynomial' in file:
                    my_marker = "o"
                elif 'samplerWith' in file:
                    my_marker = "^"
                else:
                    my_marker = "*"
                plt.plot(FW_iterations[0::n], objectives[0::n], marker=my_marker, label=my_label)

            else:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form (key, track[key][0],
                # track[key][1], multilinear_relaxation(newProblem.utility_function, track[key][1]), args.estimator,
                # args.samples)
                # track = result[0]
                # utility_function = result[1]
                solutions = []  # fractional vectors y
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                # time = []  # time it took to compute the fractional vector y
                FW_iterations = []
                degree = []  # degree of the polynomial estimator
                center = []  # point where the polynomial estimator is centered
                samples = []  # number of samples used in the sampler estimator
                for item in result:
                    FW_iterations.append(item[0])
                    solutions.append(item[2])
                    objectives.append(item[3])
                my_label = file
                if 'polynomial' in file:
                    my_marker = "o"
                elif 'samplerWith' in file:
                    my_marker = "^"
                else:
                    my_marker = "*"
                plt.plot(FW_iterations[0::n], objectives[0::n], marker=my_marker, label=my_label)
        title_str = path.split("/")[-1].split("_")
        plt.title("Selection of a subset of k = " + str(title_str[1]) + " with " + str(title_str[2]) + " FW iterations")
        plt.xlabel("Iterations")
        plt.ylabel("f^(y)")
        plt.legend()  # fontsize='xx-small')
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '.pdf', bbox_inches="tight")

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
        plt.legend()  # fontsize='xx-small')
        plt.show()
        plt.savefig('seeds.pdf', bbox_inches="tight")

    elif args.type == 'PARETO':
        plt.figure()
        # poly_time = []
        # samp_time = []
        # swd_time = []
        # poly_utility = []
        # samp_utility = []
        # swd_utility = []
        poly_dict = dict()
        samp_dict = dict()
        swd_dict = dict()
        for file in files:
            if "backup" not in file:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form
                # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                # track[key][1]), args.estimator, args.samples)
                if result[-1][4] == 'polynomial':
                    # poly_time.append(result[-1][1])
                    # #print('\n' + str(poly_time))
                    # poly_utility.append(result[-1][3])
                    poly_dict[result[-1][1]] = result[-1][3]
                    poly_label = 'POLY'
                    poly_marker = 'x'
                elif result[-1][4] == 'sampler':
                    # print('\n' + str(result[-1][4]))
                    # samp_time.append(result[-1][1])
                    # samp_utility.append(result[-1][3])
                    samp_dict[result[-1][1]] = result[-1][3]
                    samp_label = 'SAMP'
                    samp_marker = '^'
                else:
                    # print('\n' + str(result[-1][4]))
                    # swd_time.append(result[-1][1])
                    # swd_utility.append(result[-1][3])
                    swd_dict[result[-1][1]] = result[-1][3]
                    swd_label = 'SWD'
                    swd_marker = 'o'
            else:
                problem_file = path.replace('/', '_') \
                    .replace('_100_FW', '') \
                    .replace('results_continuous_greedy_', 'problems/')
                problem = load(problem_file)
                result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                file_name = file.split("_")
                track = result[1]
                max_key = max(track.iterkeys())
                time = []  # time it took to compute the fractional vector y
                # FW_iterations = []
                objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                if file_name[0] == 'polynomial':
                    poly_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                    poly_label = 'POLY'
                    poly_marker = 'x'
                elif file_name[0] == 'sampler':
                    samp_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                    samp_label = 'SAMP'
                    samp_marker = '^'
                else:
                    swd_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                    swd_label = 'SWD'
                    swd_marker = 'o'
        poly_lists = sorted(poly_dict.items())
        try:
            poly_time, poly_utility = zip(*poly_lists)
            plt.plot(poly_time, poly_utility, marker=poly_marker, markersize=8, label=poly_label, linestyle='dashed')
        except ValueError:
            pass
        samp_lists = sorted(samp_dict.items())
        try:
            samp_time, samp_utility = zip(*samp_lists)
            plt.plot(samp_time, samp_utility, marker=samp_marker, markersize=8, label=samp_label, linestyle='dashdot')
        except ValueError:
            pass
        swd_lists = sorted(swd_dict.items())
        try:
            swd_time, swd_utility = zip(*swd_lists)
            plt.plot(swd_time, swd_utility, marker=swd_marker, markersize=8, label=swd_label, linestyle='dashdot')
        except ValueError:
            pass
        plt.title("Comparison of estimators")
        plt.xlabel("time spent (seconds)")
        plt.ylabel(r'$f (\mathbf{y})$')  # , fontsize=12)
        plt.legend()  # size='large')
        plt.show()
        output_dir = 'results/plots/' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_pareto.pdf', bbox_inches="tight")

    elif args.type == 'PARETOLOG':
        plt.figure()
        poly_dict = dict()
        samp_dict = dict()
        swd_dict = dict()
        y_dict = dict()
        for file in files:
            if "backup" not in file:
                result = load(path + '/' + file)  # result is a file with a list with lines in the form
                # (key, track[key][0], track[key][1], multilinear_relaxation(newProblem.utility_function,
                # track[key][1]), args.estimator, args.samples)
                # track[key][1] is y
                if result[-1][4] == 'polynomial':
                    # result[-1][1] is running time
                    # result[-1][3] is utility aka f(y)
                    # time.append(datetime.timedelta(seconds=item[1]))  # datetime.datetime.(item[1])
                    poly_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100  # / np.sqrt(201)) * 100 # / np.log(2)) * 100
                    y = result[-1][2]
                    sys.stderr.write("\nPOLY of degree " + str(result[-1][5]) + " results in \ny = " + str(y))
                    selection = sorted(y.values(), reverse=True)
                    indices = sorted(range(1, len(y.values()) + 1), key=lambda i: y.values()[i - 1],
                                     reverse=True)
                    selection = set(indices[:k])
                    sys.stderr.write("\nThis y selects " + str(selection))
                    y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                    poly_label = 'POLY'
                    poly_marker = 'x'
                elif result[-1][4] == 'sampler':
                    # result[-1][1] is running time
                    # result[-1][3] is utility aka f(y)
                    samp_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100  # / np.sqrt(201)) * 100  # / np.log(2)) * 100
                    y = result[-1][2]
                    sys.stderr.write("\nSAMP with " + str(result[-1][5]) + " samples results in \ny = " + str(y))
                    selection = sorted(y.values(), reverse=True)
                    indices = sorted(range(1, len(y.values()) + 1), key=lambda i: y.values()[i - 1],
                                     reverse=True)
                    selection = set(indices[:k])
                    sys.stderr.write("\nThis y selects " + str(selection))
                    y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                    samp_label = 'SAMP'
                    samp_marker = '^'
                else:
                    # result[-1][1] is running time
                    # result[-1][3] is utility aka f(y)
                    swd_dict[result[-1][1]] = result[-1][3]  # / np.log(2)) * 100 # / np.sqrt(201)) * 100 #
                    y = result[-1][2]
                    sys.stderr.write("\nSWD with " + str(result[-1][5]) + " samples results in \ny = " + str(y))
                    selection = sorted(y.values(), reverse=True)
                    indices = sorted(range(1, len(y.values()) + 1), key=lambda i: y.values()[i - 1],
                                     reverse=True)
                    selection = set(indices[:k])
                    sys.stderr.write("\nThis y selects " + str(selection))
                    y_dict[(result[-1][4], result[-1][5])] = (y, selection)  # y
                    swd_label = 'SWD'
                    swd_marker = 'o'
        # sys.stderr.write("\ny_dict = " + str(y_dict))
        for est in y_dict:
            if est != ('polynomial', 1):
                common = y_dict[est][1].intersection(y_dict[('polynomial', 1)][1])
                different = y_dict[est][1].difference(y_dict[('polynomial', 1)][1])
                sys.stderr.write("\n" + str(est[0]) + " estimator with " + str(est[1]) + " chooses the following "
                                 "elements in common with the polynomial estimator of degree 1 \n" + str(common) +
                                 "\nand the following elements differently \n" + str(different))
            else:
                pass
                # problem_file = path.replace('/', '_') \
                #     .replace('_100_FW', '') \
                #     .replace('results_continuous_greedy_', 'problems/')
                # problem = load(problem_file)
                # result = load(path + '/' + file)  # result is a list in the format [y, track, bases]
                # file_name = file.split("_")
                # track = result[1]
                # max_key = max(track.iterkeys())
                # time = []  # time it took to compute the fractional vector y
                # # FW_iterations = []
                # objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
                # if file_name[0] == 'polynomial':
                #     poly_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                #     poly_label = 'POLY'
                #     poly_marker = 'x'
                # elif file_name[0] == 'sampler':
                #     samp_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                #     samp_label = 'SAMP'
                #     samp_marker = '^'
                # else:
                #     swd_dict[track[max_key][0]] = problem.utility_function(track[max_key][1])
                #     swd_label = 'SWD'
                #     swd_marker = 'o'
        poly_lists = sorted(poly_dict.items())
        try:
            poly_time, poly_utility = zip(*poly_lists)
            print(poly_time, poly_utility, 'POLY')
            plt.semilogx(poly_time, poly_utility, marker=poly_marker, markersize=8, label=poly_label, linestyle='dashed')
        except ValueError:
            pass
        samp_lists = sorted(samp_dict.items())
        try:
            samp_time, samp_utility = zip(*samp_lists)
            print(samp_time, samp_utility, 'SAMP')
            plt.semilogx(samp_time, samp_utility, marker=samp_marker, markersize=8, label=samp_label, linestyle='dashdot')
        except ValueError:
            pass
        swd_lists = sorted(swd_dict.items())
        try:
            swd_time, swd_utility = zip(*swd_lists)
            print(swd_time, swd_utility, 'SWD')
            plt.semilogx(swd_time, swd_utility, marker=swd_marker, markersize=8, label=swd_label, linestyle='dashdot')
        except ValueError:
            pass
        plt.title("Comparison of estimators")
        plt.xlabel("time spent (seconds)")
        # plt.ylabel(r"$\frac{f(\mathbf{y})}{\max{f(\mathbf{y})}} \times 100$")  # , fontsize=12)
        plt.ylabel(r"$f(\mathbf{y})$")  # , fontsize=12)
        plt.legend()  # fontsize='large')
        # axes = figure().add_subplot(111)
        # a = axes.get_xticks().tolist()
        # a[1] = 'change'
        # axes.set_xticklabels(a)
        # plt.show()
        # import numpy as np
        # import datetime
        # import matplotlib.pyplot as plt
        # import matplotlib.dates as mdates
        #
        # days = [2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16]
        #
        # time_list = [datetime.timedelta(0, 23820), datetime.timedelta(0, 27480),
        #              datetime.timedelta(0, 28500), datetime.timedelta(0, 24180),
        #              datetime.timedelta(0, 27540), datetime.timedelta(0, 28920),
        #              datetime.timedelta(0, 28800), datetime.timedelta(0, 29100),
        #              datetime.timedelta(0, 29100), datetime.timedelta(0, 24480),
        #              datetime.timedelta(0, 27000)]
        #
        # # specify a date to use for the times
        # zero = datetime.datetime(0, 1, 1)
        # time = [zero + t for t in time_list]
        # # convert datetimes to numbers
        # zero = mdates.date2num(zero)
        # time = [t - zero for t in mdates.date2num(time)]
        #
        # f = plt.figure()
        # ax = f.add_subplot(1, 1, 1)
        #
        # ax.bar(days, time, bottom=zero)
        # ax.yaxis_date()
        # ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        #
        # # add 10% margin on top (since ax.margins seems to not work here)
        # ylim = ax.get_ylim()
        # ax.set_ylim(None, ylim[1] + 0.1 * np.diff(ylim))
        #
        # plt.show()
        # ax = plt.gca()
        # for item in ax.get_xticks().tolist():
        #     print(item)
        # labels = [datetime.timedelta(seconds=int(item.get_text())) for item in ax.get_xticks().tolist()]
        # ax.set_xticklabels(labels)
        # plt.show()
        output_dir = 'results/plots/' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '/paretolog.pdf', bbox_inches="tight")

    elif args.type == 'CONV_TEST':
        plt.figure()
        poly_dict = dict()
        samp_dict = dict()
        swd_dict = dict()
        for file in files:
            result = load(path + '/' + file)  # result is a list in the format [elapsed_time, args.degree,
            # poly_estimation]
            file_name = file.split("_")
            # track = result[1]
            # max_key = max(track.iterkeys())
            time = []  # time it took to compute the fractional vector y
            # FW_iterations = []
            estimates = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
            if file_name[0] == 'polynomial':
                poly_dict[result[0]] = result[2]
                poly_label = 'POLY'
                poly_marker = 'x'
            elif file_name[0] == 'sampler':
                samp_dict[result[0]] = result[2]
                samp_label = 'SAMP'
                samp_marker = '^'
            else:
                swd_dict[result[0]] = result[2]
                swd_label = 'SWD'
                swd_marker = 'o'
        poly_lists = sorted(poly_dict.items())
        try:
            poly_time, poly_utility = zip(*poly_lists)
            plt.semilogx(poly_time, poly_utility, marker=poly_marker, markersize=8, label=poly_label,
                         linestyle='dashed')
        except ValueError:
            pass
        samp_lists = sorted(samp_dict.items())
        try:
            samp_time, samp_utility = zip(*samp_lists)
            plt.semilogx(samp_time, samp_utility, marker=samp_marker, markersize=8, label=samp_label,
                         linestyle='dashdot')
        except ValueError:
            pass
        swd_lists = sorted(swd_dict.items())
        try:
            swd_time, swd_utility = zip(*swd_lists)
            plt.semilogx(swd_time, swd_utility, marker=swd_marker, markersize=8, label=swd_label, linestyle='dashdot')
        except ValueError:
            pass
        plt.title("Comparison of estimators")
        plt.xlabel("time spent (seconds)")
        plt.ylabel(r'$\hat{G} (\mathbf{y})$')  # , fontsize=12)
        plt.legend()  # fontsize='large')
        plt.show()
        output_dir = 'results/plots/tests/conv_tests/' + path.replace("results/convergence_test/problems", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '_conv_test.pdf', bbox_inches="tight")

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
        # plt.legend()  # fontsize='xx-small')
        # plt.show()
        # output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # # plt.savefig(output_dir + '_time.pdf', bbox_inches="tight")
        # # plt.savefig(output_dir + '.pdf', bbox_inches="tight")
        # plt.savefig(output_dir + '_logtime.pdf', bbox_inches="tight")

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
    # plt.savefig('results/plots/GreedyAlgorithmEpinions100IM3.pdf')
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
    # plt.savefig('results/plots/Comparisons_on_test_graphs.pdf')
    # for file in os.listdir("results/continuous_greedy"):
    #    if "IMtest_casepolynomial_100_FW_degree_4_around_0.0" in file or 'IMtest_casesampler_100_FW_100samples' in file:

         # sys.stderr.write("cont_greedy track is: " + str(cont_greedy_track))
    #        utility = []
    #        cardinality = []
    #        for item in cont_greedy_track:
    #            utility.append(item[2])
    #            cardinality.append(item[0])
    #        plt.plot(cardinality, utility, "^", label=file)
    # plt.legend()  # fontsize='xx-small')
    # plt.savefig('results/plots/sample_vs_poly_on_test_graphs.pdf')


    ## UNCOMMENT HERE FOR COMPARING POLYNOMIAL ESTIMATORS CENTERED AROUND DIFFERENT POINTS #
    # plt.figure()
    # plt.plot(degrees1, utility1, "g^", label="Polynomial around 0.0", markersize=5)
    # plt.plot(degrees2, utility2, "ro", label="Polynomial around 0.5", markersize=5)
    # # plt.plot(time3, utility3, "bx", label="Polynomial", markersize=5)
    # plt.plot(degrees2, out, label="Multilinear Relaxation")
    # plt.legend()  # fontsize='xx-small')
    # plt.xlabel('Degree')
    # plt.ylabel('Utility')
    # plt.title('Comparison of polynomial estimators around 0.0 and 0.5')
    # plt.savefig('results/plots/1Graph_y_05_DegreeVSUtilityOnPolynomials00and05.pdf')
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
    # plt.legend()  # size='xx-small')
    # plt.xlabel('Iterations')
    # plt.ylabel('Multilinear Relaxation')
    # plt.savefig('results/plots/multivsiterationsDR0.pdf')
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
    #plt.savefig('results/plots/2ndDegreePolynomialEstimatorAround00.pdf')

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
    #plt.savefig('IterVSTime.pdf')
