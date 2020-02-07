import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from matplotlib.transforms import Bbox
import numpy as np
from matplotlib.dates import date2num
import datetime
import os



if __name__ == "__main__":

    time_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_time", "r").read())
    utility_ax = eval(open("results/IM_on_smaller_Epinions_dataset_with10seeds_polynomialestimator_300_FW_2th_degree_around_0.0_utilities", "r").read())
    #sampler_obj = eval(open('sampler_obj.txt', 'r').read())
    #iterations1 = list(range(1, len(sampler_obj) + 1))

    #poly_obj = eval(open('poly_obj.txt', 'r').read())
    #iterations2 = list(range(1, len(poly_obj) + 1))

    plt.plot(time_ax, utility_ax, label = '2nd degree Polynomial Estimator around 0.0')
    #plt.plot(iterations1, sampler_obj, 'g^', label = 'Sampler Estimator')
    plt.xlabel('Time')
    plt.ylabel('Utility')
    plt.legend()
    plt.savefig('results/plots/2ndDegreePolynomialEstimatorAround00.png')

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
