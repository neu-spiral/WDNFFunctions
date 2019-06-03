import matplotlib.pyplot as plt
import argparse
from matplotlib.transforms import Bbox
import numpy as np
from matplotlib.dates import date2num
import datetime
import os



colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
hatches = ['////', '/', '\\', '\\\\', '-', '--', '+', '']


Algorithms = ['SE-CU','CU-SE','SE-Greedy','CG-RS500','CGT1','CGT2']
graph2lbl =  {'erdos_renyi':'ER','erdos_renyi2':'ER-20Q','hypercube':'HC','hypercube2':'HC-20Q'}
    
Graphs = ['erdos_renyi','erdos_renyi2','star','hypercube', 'hypercube2','dtelekom','abilene','geant']


def bar_ex1(DICS,outfile):
    def form_vals(dic):
        vals = []
        lbls = []
        for key in Graphs:
            vals.append(dic[key])
            if key in graph2lbl:
                key = graph2lbl[key]
            lbls.append(key)
        return vals,lbls

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    width = 1
    N = len(Graphs)
    numb_bars = len(Algorithms)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    RECTS = []
    i = 0
    for alg in Algorithms:
        RECTS+= ax.bar(ind+i*width, form_vals(DICS[alg])[0], align='center',width=width, color = colors[i], hatch = hatches[i],label=alg,log=True)
        i+=1
    '''Set legend'''
    LGD = ax.legend([alg for alg in Algorithms], ncol=len(Algorithms),borderaxespad=0.,loc=3, bbox_to_anchor=(0., 1.02, 1., .102),fontsize=15,mode='expand')

    lbls =  form_vals(DICS[alg])[1]
    
    ax.set_xticks(ind+width*3) 
    ax.set_xticklabels(tuple(lbls),fontsize = 16)

    if args.mode2 == 'OBJ':
        y_label = args.mode
    else:
        y_label = 'Time'
    ax.set_ylabel(y_label,fontsize=22)
    plt.yticks(fontsize = 18)
    plt.xlim([ind[0]-width,ind[-1]+len(Algorithms)*width])
    '''set axis range'''
    if args.mode2 == 'TIME':
        plt.ylim(0.003,80000)
    else:
        plt.ylim(50,10000)

    fig.savefig(outfile+".pdf",format='pdf', bbox_extra_artists=(LGD,) )

    plt.show() 

def read_file(fname,normalize=0):
    f = open(fname,'r')
    l= eval(f.readline())
    f.close()
    (Time, OBJ) = l[-1] 
    return {"TIME":Time,"OBJ":OBJ}

def CG_readfile(CG_file,rounded_file):
    dic_CG = read_file(CG_file)
    dic_round = read_file(rounded_file)
    return {"TIME":dic_CG["TIME"]+dic_round["TIME"],"OBJ":dic_round["OBJ"]}


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Generate bar plots comparing different algorithms.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode',default='Cost', type=str,choices=['Gain','Cost'],help='Determine whether to plot gain or cost')
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--mode2',default='OBJ', type=str,choices=['OBJ','TIME'],help='Determine whether to plot time or objective')
    args = parser.parse_args()
    
    title = "Power law arrival distribution"
    DICS = {"SE-CU":{}, 'CU-SE':{}, 'CG-RS500':{}, 'CGT1':{}, 'CGT2':{}, 'SE-Greedy':{}}
    for graph in Graphs:
       
        if graph == "erdos_renyi2" or graph == "hypercube2":
            problem_instance = "problem_%s_1000demands_100catalog_size_2mincap_2maxcap_100_powerlaw" %(graph[0:-1]) + "_rate1.0_20qnodes"
        else: 
            problem_instance = "problem_%s_1000demands_100catalog_size_2mincap_2maxcap_2_100_powerlaw" %(graph) + "_rate1.0_4qnodes"
    
        random_1 = "RANDOM/" + problem_instance + "-service_first"
        random_2  = "RANDOM/"+ problem_instance +"-item_first"
        greedy_f = "GREEDY/" + problem_instance
        CG_1 = "OUTPUT/"+ problem_instance + "-500-samples-1000-iters" 
        round_1 = "ROUNDED/"+ problem_instance + "-500-samples-1000-iters"
        CG_2 = "OUTPUT/"+ problem_instance + "-taylor-k-1-1000-iters"
        round_2 = "ROUNDED/"+ problem_instance + "-taylor-k-1-1000-iters"
        CG_3 = "OUTPUT/"+ problem_instance + "-taylor-k-2-1000-iters"
        round_3 = "ROUNDED/"+ problem_instance + "-taylor-k-2-1000-iters"

        mode2 = args.mode2
        if mode2 == 'TIME':
            C0 = 0
            SGN = +1
        else:
            if args.mode == 'Cost':
                f_c0 = open(random_1, 'r')
                OBJ0 = eval(f_c0.readline())[0]
                f_c0.close()
                C0 = OBJ0
                SGN = -1

            else:
                C0 = 0
                SGN = +1

        if args.normalize:
            RAN_VAL = C0 + SGN*read_file(random_1)[mode2]
        else:
            RAN_VAL = 1

        DICS['SE-CU'][graph] = (C0 + SGN*read_file(random_1)[mode2])/RAN_VAL
        DICS['CU-SE'][graph] = (C0 + SGN * read_file(random_2)[mode2]) / RAN_VAL
        DICS['CG-RS500'][graph] = (C0 + SGN*CG_readfile(CG_1,round_1)[mode2] )/RAN_VAL
        DICS['CGT1'][graph] = (C0 + SGN * CG_readfile(CG_2, round_2)[mode2]) / RAN_VAL
        DICS['CGT2'][graph] = (C0 + SGN*CG_readfile(CG_3,round_3)[mode2])/RAN_VAL
        DICS['SE-Greedy'][graph] = (C0 +SGN*read_file(greedy_f)[mode2])/RAN_VAL

    print DICS

    if args.mode2 == 'TIME':
        dir = "TIME/"
    else:
        dir = "OBJ/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    outfile = dir + "problem_1000demands_300catalog_size_mincap_2maxcap_2_100_powerlaw"
    bar_ex1(DICS,outfile)
