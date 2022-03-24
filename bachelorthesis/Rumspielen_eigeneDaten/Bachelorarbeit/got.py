import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd
from numpy import transpose
import numpy as np


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
    bars = 80
    fig1, axes = plt.subplots(2, 2)
    fig1.set_size_inches(9, 7)

    # degree_centralities
    degree = calculate_degree_centrality(G)
    #print("degree looks like", degree)
    dist_degree, x_degree, bar_degree = generate_denisity(bars, degree)
    #print(dist_degree,x_degree,bar_degree)
    axes[0][0].bar(x_degree, np.array(dist_degree), bar_degree, align='edge', label="distribution")
    axes[0][0].set_title("distribution over degree-centrality")
    axes[0][0].set_ylabel("amount of nodes (total nodes = " + str(len(degree)) + ")")
    axes[0][0].legend()

    # closness_centralities
    closeness = calculate_closness_centrality(G)
    #print("closeness looks like", closeness)
    dist_closeness, x_closeness, bar_closeness = generate_denisity(bars, closeness)
    axes[0][1].bar(x_closeness, np.array(dist_closeness), bar_closeness, align='edge', label="distribution")
    axes[0][1].set_title("distribution over closeness-centrality")
    axes[0][1].set_ylabel("amount of nodes (total nodes = " + str(len(closeness)) + ")")
    axes[0][1].legend()

    # between_centralities
    between = calculate_between_centrality(G)
    #print("betweenness looks like", between)
    dist_between, x_between, bar_between = generate_denisity(bars, between)
    axes[1][0].bar(x_between, np.array(dist_between), bar_between, align='edge', label="distribution")
    axes[1][0].set_title("distribution over betweenness-centrality")
    axes[1][0].set_ylabel("amount of nodes (total nodes = " + str(len(between)) + ")")
    axes[1][0].legend()

    # distribution n times degree_centralities
    eigenvector = calculate_eigenvector_centrality(G)
    #print("eigenvector looks like", eigenvector)
    dist_eigenvector, x_eigenvector, bar_eigenvector = generate_denisity(bars, eigenvector)
    axes[1][1].bar(x_eigenvector, np.array(dist_eigenvector), bar_eigenvector, align='edge', label="distribution")
    axes[1][1].set_title("distribution over eigenvector-centrality")
    axes[1][1].set_ylabel("amount of nodes (total nodes = " + str(len(eigenvector)) + ")")
    axes[1][1].legend()
    plt.show()
    plt.savefig("outputFacebookPoliticsDistribution.png")

    pos = nx.spring_layout(G, seed=196900)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=range(len(G)), labels={node: node for node in G.nodes()}, font_size=2, cmap=plt.cm.tab10,
            node_size=15, edge_color="#D4D5CE", width=0.4, linewidths=0.4)
    plt.show()
    plt.savefig("outputFacebookPolitics.png")

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


df = pd.read_csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/facebook_combined.csv", sep=r',')
df.head()
df1 = df[['Source', 'Target']]


G = nx.from_pandas_edgelist(df1, 'Source', 'Target')
print("average shortest path:", nx.average_shortest_path_length(G))
A = nx.to_numpy_matrix(G)
#print("Adjacency matrix looks like", A)
generate_got_density(G)
