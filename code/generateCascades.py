from networkx import DiGraph
from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import logging
import numpy as np
import os
import pickle
import sys


if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)
    logging.info('Reading graph...')
    # G = read_edgelist("soc-Epinions1.txt", comments='#', create_using=DiGraph(), nodetype=int)
    # degrees = dict(list(G.out_degree(G.nodes())))
    # descending_degrees = sorted(degrees.values(), reverse=True)
    # indices = sorted(range(1, len(degrees.values()) + 1), key=lambda k: degrees.values()[k - 1], reverse=True)
    # top10000_indices = indices[:10001]
    # top100_indices = indices[:100]
    # G = G.subgraph(top100_indices)
    G = DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    # G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # G.add_edges_from([(1, 2), (2, 3), (4, 5), (4, 6), (6, 3), (10, 9), (10, 3), (7, 8)])
    # graphs = [new_graph]
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('...done. Created a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    logging.info('Creating cascades...')
    # p = 0.2
    # numberOfCascades = 10
    # newG = DiGraph()
    # newG.add_nodes_from(G.nodes())
    # graphs = [newG] * numberOfCascades
    # os.mkdir("edge_lists")
    # for cascade in range(numberOfCascades):
    #     choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
    #     chosen_edges = np.extract(choose, G.edges())
    #     chosen_edges = zip(chosen_edges[0::2], chosen_edges[1::2])
    #     graphs[cascade].add_edges_from(chosen_edges)
    #     sys.stderr.write("edge list of cascade #" + str(cascade) + " : " + str(graphs[cascade].edges) + '\n')
    #     logging.info('Created cascade %d.' % cascade)
    #    write_edgelist(graphs[cascade], "edge_lists/cascade" + str(cascade))
    # logging.info('...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p))

    # graphs = []
    # for cascade in os.listdir("edge_lists"):
        # logging.info('Creating cascade %d...' % ())
        # G = read_edgelist("edge_lists/" + cascade, create_using=DiGraph(), nodetype=int)
        # graphs.append(G)

    graphs = [DiGraph()]
    graphs[0].add_nodes_from([1, 2, 3])
    graphs.append(DiGraph())
    graphs[1].add_nodes_from([1, 2, 3])
    graphs.append(DiGraph())
    graphs[2].add_nodes_from([1, 2, 3])
    graphs.append(DiGraph())
    graphs[3].add_nodes_from([1, 2, 3])
    graphs.append(DiGraph())
    graphs[4].add_nodes_from([1, 2, 3])
    graphs.append(DiGraph())
    graphs[5].add_nodes_from([1, 2, 3])
    graphs[5].add_edges_from([(2, 3)])
    graphs.append(DiGraph())
    graphs[6].add_nodes_from([1, 2, 3])
    graphs[6].add_edges_from([(2, 3)])
    graphs.append(DiGraph())
    graphs[7].add_nodes_from([1, 2, 3])
    graphs[7].add_edges_from([(2, 3)])
    graphs.append(DiGraph())
    graphs[8].add_nodes_from([1, 2, 3])
    graphs[8].add_edges_from([(2, 3)])
    graphs.append(DiGraph())
    graphs[9].add_nodes_from([1, 2, 3])
    graphs[9].add_edges_from([(1, 2), (2, 3)])
    for i in range(10):
        sys.stderr.write("edge list of graph #" + str(i) + " : " + str(graphs[i].edges()) + '\n')

    with open("datasets/mini_graphs_file", "w") as f:
        pickle.dump(graphs, f)
