from networkx import DiGraph
from networkx.readwrite.edgelist import read_edgelist
import logging
import numpy as np


if __name__ == "__main__":

    G = read_edgelist("soc-Epinions1.txt", comments='#', create_using=DiGraph(), nodetype=int)
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('...done. Created a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    logging.info('Creating cascades...')
    p = 0.02
    numberOfCascades = 1
    newG = DiGraph()
    newG.add_nodes_from(G.nodes())
    graphs = [newG] * numberOfCascades
    for cascade in range(numberOfCascades):
        choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
        chosen_edges = np.extract(choose, G.edges())
        chosen_edges = zip(chosen_edges[0::2], chosen_edges[1::2])
        graphs[cascade].add_edges_from(chosen_edges)
        logging.info('...done. Created %d cascades with %d infection probability.' % (numberOfCascades, p))

    f = open("independentCascades.txt", "w")
    f.write(str(graphs))
    f.close()