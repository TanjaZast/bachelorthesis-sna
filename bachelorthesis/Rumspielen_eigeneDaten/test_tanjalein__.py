import networkx as nx
import numpy as np
import pylab as plt
from astropy.table import Table
from networkx.drawing.nx_agraph import graphviz_layout
from numpy import transpose
from tabulate import tabulate

G = nx.Graph()
G.add_edges_from(
[(3, 1), (3, 4), (3, 6), (3, 5), (4, 5), (4, 6), (6, 5), (1,2), (2,7), (7,8), (8,1), (7,1), (8, 2), (3,9), (9,10), (10,11), (9, 11)])
nx.draw_networkx(G, node_size = 300, font_size=10, node_color= 'lightblue')

degree = nx.degree_centrality(G)
listDegree = list(degree.items())
arrayDegree = np.array(listDegree)
transposedDegree = transpose(arrayDegree)
degree_centrality = transposedDegree[1]

close = nx.closeness_centrality(G)
listClose = list(close.items())
arrayClose = np.array(listClose)
transposedClose = transpose(arrayClose)
close_centrality = transposedClose[1]

between = nx.betweenness_centrality(G)
listBetween = list(between.items())
arrayBetween = np.array(listBetween)
transposedBetween = transpose(arrayBetween)
between_centrality = transposedBetween[1]

eigen = nx.eigenvector_centrality(G)
listEigen = list(eigen.items())
arrayEigen = np.array(listEigen)
transposedEigen = transpose(arrayEigen)
eigen_centrality = transposedEigen[1]

def tables(degree, closeness, between, eigen):

    node = nx.degree_centrality(G)
    node = list(node.items())
    node_array = np.array(node)
    node = transpose(node_array)
    node = node[0]

    t = Table([node, degree, closeness, between, eigen], names=('Nodes', 'Degree', 'closeness', 'between', 'eigen'))
    #print(t)
    print(tabulate(t, tablefmt="latex"))

tables(degree_centrality, close_centrality, between_centrality, eigen_centrality)

plt.show()
