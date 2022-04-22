from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd
from astropy.table import Table
from numpy import transpose
import numpy as np
from tabulate import tabulate
import seaborn as sns


def generate_denisity(bars, centralities):

    centralities = np.array(centralities)

    #initializing distribution
    minBarPlot =  np.min(centralities)
    maxBarPlot =  np.max(centralities)
    dist = np.zeros(bars)
    intv = maxBarPlot - minBarPlot
    x = np.linspace(np.min(centralities),np.max(centralities),bars)
    barsize = ((maxBarPlot - minBarPlot) / bars) * 0.8

    for d in centralities:
        dist[int(((d - minBarPlot) * bars) / intv) - 1] = dist[int(((d - minBarPlot) * bars) / intv) - 1] + 1

    return [dist, x, barsize]


def generate_got_density(G):
    # generate big graph with all individual matrices combined

    #print(big_graph.shape)
    bars = 100
    fig1, axes = plt.subplots(2, 2)
    fig1.set_size_inches(9, 7)

    # degree_centralities
    degree = calculate_degree_centrality(G)
    nodes = calculate_nodes(G)
    print("degree looks like", degree)
    sns.distplot(np.log(degree), ax=axes[0][0], bins=bars, label='distribution')
    sns.rugplot(np.log(degree), clip_on=False, alpha=0.01, height=-0.02, ax=axes[0][0])
    axes[0][0].set_ylabel("amount of nodes (total nodes = " + str((len(nodes))) + ")")
    axes[0][0].set_title("distribution over degree-centrality")
    axes[0][0].legend()

    # closness_centralities
    closeness = calculate_closness_centrality(G)
    print("closeness looks like", closeness)
    sns.distplot(closeness, ax=axes[0][1], bins=bars, label='distribution')
    sns.rugplot(closeness, clip_on=False, alpha=0.01, height=-0.02, ax=axes[0][1])
    axes[0][1].set_ylabel("amount of nodes")
    axes[0][1].set_title("distribution over closeness-centrality")
    axes[0][1].legend()

    # between_centralities
    between = calculate_between_centrality(G)
    print("betweenness looks like", between)
    between = np.array(between)
    between = between + 0.0000000000001
    sns.distplot(np.log(between), ax=axes[1][0], bins=bars, label='distribution')
    sns.rugplot(np.log(between), clip_on=False, alpha=0.01, height=-0.02, ax=axes[1][0])
    axes[1][0].set_ylabel("amount of nodes")
    axes[1][0].set_title("distribution over betweenness-centrality")
    axes[1][0].legend()

    # distribution n times degree_centralities
    eigenvector = calculate_eigenvector_centrality(G)
    print("eigenvector looks like", eigenvector)
    eigenvector = np.array(eigenvector)
    eigenvector = eigenvector + 0.0000000000001
    sns.distplot(np.log(eigenvector), ax=axes[1][1], bins=bars, label='distribution')
    sns.rugplot(np.log(eigenvector), clip_on=False, alpha=0.01, height=-0.02, ax=axes[1][1])
    axes[1][1].set_ylabel("amount of nodes")
    axes[1][1].set_title("distribution over eigenvector-centrality")
    axes[1][1].legend()
    plt.savefig('/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/Plots/GOT-Distribution.png')
    plt.show()


    pos = nx.spring_layout(G, seed=196900)
    plt.figure(figsize=(10, 8))
    #labels={node: node for node in G.nodes()}
    #font_size = 2
    nx.draw(G, pos, node_color=range(len(G)), cmap=plt.cm.tab10,
            node_size=15, edge_color="#D4D5CE", width=0.4, linewidths=0.4)
    plt.savefig('/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/Plots/GOT-Plot.png')
    plt.show()

    Gz0 = nodes
    Gz1 = degree
    Gz2 = closeness
    Gz3 = between
    Gz4 = eigenvector
    t = Table([Gz0, Gz1, Gz2, Gz3, Gz4], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvector'))
    # print(t)
    #print(tabulate(t, tablefmt="latex"))

    i = 0
    lenght = []
    for clq in nx.clique.find_cliques(G):
        i = i + 1
        lenght.append(len(clq))
    print('amount of cliques', i)
    print('biggest clique', max(lenght))
    node = G.number_of_nodes()
    print('number of nodes', node)
    edge = G.number_of_edges()
    print('number of edges', edge)


def calculate_degree_centrality(G):

    degree = nx.degree_centrality(G)
    listDegree = list(degree.items())
    arrayDegree = np.array(listDegree)
    transposed = transpose(arrayDegree)
    degree_centrality = transposed[1]
    degree_centrality = np.array(degree_centrality)
    degree_centrality = degree_centrality.astype(float)
    return degree_centrality

def calculate_closness_centrality(G):

    closeness = nx.closeness_centrality(G)
    listCloseness = list(closeness.items())
    arrayCloseness = np.array(listCloseness)
    transposed = transpose(arrayCloseness)
    close_centrality = transposed[1]
    close_centrality = np.array(close_centrality)
    close_centrality = close_centrality.astype(float)
    #print('closeness centrality looks like', close_centrality)
    return close_centrality

def calculate_nodes(G):
    degree = nx.degree_centrality(G)
    listDegree = list(degree.items())
    arrayDegree = np.array(listDegree)
    transposed = transpose(arrayDegree)
    nodes = transposed[0]
    nodes = np.array(nodes)
    return nodes

def calculate_between_centrality(G):

    between = nx.betweenness_centrality(G)
    listBetween = list(between.items())
    arrayBetween = np.array(listBetween)
    transposed = transpose(arrayBetween)
    between_centrality = transposed[1]
    between_centrality = np.array(between_centrality)
    between_centrality = between_centrality.astype(float)
    #print('betweenness centrality looks like', between_centrality)
    return between_centrality

def calculate_eigenvector_centrality(G):

    eigenvector = nx.eigenvector_centrality_numpy(G)
    listEigenvector = list(eigenvector.items())
    arrayEigenvector = np.array(listEigenvector)
    transposed = transpose(arrayEigenvector)
    eigenvector_centrality = transposed[1]
    eigenvector_centrality = np.array(eigenvector_centrality)
    eigenvector_centrality = eigenvector_centrality.astype(float)
   #print('closeness centrality looks like', eigenvector_centrality)
    return eigenvector_centrality


df = pd.read_csv("/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/CSV/asoiaf-all-edges.csv", sep=r',')
df.head()
df1 = df[['Source', 'Target']]


G = nx.from_pandas_edgelist(df1, 'Source', 'Target')
#print("average shortest path:", nx.average_shortest_path_length(G))
A = nx.to_numpy_matrix(G)
#print("Adjacency matrix looks like", A)
generate_got_density(G)
