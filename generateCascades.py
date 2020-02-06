from networkx import DiGraph
from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import logging
import numpy as np
import os
import pickle


if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)
    #logging.info('Reading graph...')
    #G = read_edgelist("soc-Epinions1.txt", comments='#', create_using=DiGraph(), nodetype=int)
    #degrees = dict(list(G.out_degree(G.nodes())))
    #descending_degrees = sorted(degrees.values(), reverse = True)
    #indices = sorted(range(1, len(degrees.values()) + 1), key = lambda k: degrees.values()[k - 1], reverse = True)
    #top10000_indices = indices[:10001]
    #G = G.subgraph(top10000_indices)
    #numOfNodes = G.number_of_nodes()
    #numOfEdges = G.number_of_edges()
    #logging.info('...done. Created a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    #logging.info('Creating cascades...')
    #p = 0.02
    #numberOfCascades = 1000
    #newG = DiGraph()
    #newG.add_nodes_from(G.nodes())
    #graphs = [newG] * numberOfCascades
    #os.mkdir("edge_lists")
    #for cascade in range(numberOfCascades):
    #    choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
    #    chosen_edges = np.extract(choose, G.edges())
    #    chosen_edges = zip(chosen_edges[0::2], chosen_edges[1::2])
    #    graphs[cascade].add_edges_from(chosen_edges)
    #    write_edgelist(graphs[cascade], "edge_lists/cascade" + str(cascade))
    #logging.info('...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p))

    graphs = []
    for cascade in os.listdir("edge_lists"):
        #logging.info('Creating cascade %d...' % ())
        G = read_edgelist("edge_lists/" + cascade, create_using=DiGraph(), nodetype=int)
        graphs.append(G)
        logging.info('Created cascade %d...' % (len(graphs)))

    with open("graphs_file", "w") as f:
        pickle.dump(graphs, f)