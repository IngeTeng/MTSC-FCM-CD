
import math
import numpy as np
import networkx as nx
def construct_network(similar,t):

    S_list = []
    for i in range(similar.shape[0]):
        for j in range(similar.shape[1]):
            if  j < i :
                S_list.append(similar[i,j])
    #threshold
    L_order = sorted(S_list, reverse=True)
    position = math.ceil(len(L_order) * t)
    threshold = L_order[position]

    g = nx.Graph()
    g.add_nodes_from(range(similar.shape[0]))

    for i in range(similar.shape[0]):
        for j in range(similar.shape[1]):
            if i < j:
                if similar[i][j] >= threshold:
                    g.add_edge(i, j)
    return g
