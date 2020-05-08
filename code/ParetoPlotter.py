from helpers import load
import argparse
import datetime
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = "results/continuous_greedy/IM/epinions_50"  # "results/continuous_greedy/FL/ratings_10"
    files = os.listdir(path)
    for file in files:
        plt.figure()
        result = load(path + '/' + file)  # result is a file with lines (y, F(y), time_passed, FW_iterations,
        # estimator_type, degree, center) or (y, F(y), time_passed, FW_iterations, estimator_type, num_of_samples)
        solutions = []  # fractional vectors y
        objectives = []  # F(y) where F is the multilinear relaxation or F^(y) where F^ is the best estimator
        time = []  # time it took to compute the fractional vector y
        FW_iterations = []
        degree = []  # degree of the polynomial estimator
        center = []  # point where the polynomial estimator is centered
        samples = []  # number of samples used in the sampler estimator
        for item in result:
            solutions.append(item[0])
            objectives.append(item[1])
            time.append(np.log(item[2]))
            FW_iterations.append(item[3])
            if item[4] == 'polynomial':
                print('\n' + str(item))
                degree.append(item[5])
                center.append(item[6])
            else:
                samples.append(item[5])
        # print('time axis is: ' + str(time))
        max_objective = max(objectives)
        best_solution = solutions[objectives.index(max_objective)]
        objectives = ((np.array(objectives) - max_objective) / max_objective) + 0.06
        if degree:
            # print('degrees are' + str(degree))
            my_label = str(FW_iterations[0]) + ' FW iterations, center = ' + str(center[0])
            plt.plot(time, objectives, 's', label=my_label)
            plt.legend(fontsize='xx-small')
            axes1 = plt.gca()
            axes2 = axes1.twiny()
            axes2.set_xlim(axes1.get_xlim())
            axes2.set_xticks(time)
            axes2.set_xticklabels(degree)
            axes2.set_xlabel('degrees')
        else:
            # print('samples are' + str(samples))
            plt.plot(time, objectives, 's', label=(str(FW_iterations[0]) + ' FW iterations'))
            axes1 = plt.gca()
            axes2 = axes1.twiny()
            axes2.set_xticks(time)
            axes2.set_xticklabels(samples)
            axes2.set_xlabel('samples')
        plt.legend(fontsize='x-small')
        axes1.set_xlabel('log(time) spent to compute fractional vector solution y')
        axes1.set_ylabel('(F^(y) - F^(y*)) / F^(y*)')
        # plt.title('Comparison of different estimators')
        # for item in result:
        #     if item[4] == 'polynomial':
        #         plt.text(item[0], item[1] + 0.01, 'd=' + str(item[5]) + ', c=' + str(item[6]))
        #     else:
        #         plt.text(item[0], item[1] + 0.01, '#=' + str(item[5]))
        plt.show()
        output_dir = 'results/plots' + path.replace("results/continuous_greedy", "/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + file + '.png', bbox_inches="tight")


