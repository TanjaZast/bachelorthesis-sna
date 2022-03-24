import functools
import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import dijkstra
from matplotlib import pyplot as plt
from networkx import eigenvector_centrality, random_regular_graph, edge_betweenness_centrality, find_cliques
import networkx as nx
from numpy import transpose
from astropy.table import Table
from tabulate import tabulate


#method is an integer and
# 1 represents method
# 2

#node is the amount of nodes, probability is the probability that there is a connection between point a and b
def generate_graph(node,probability, seed, method):
    G = nx.gnp_random_graph(node, probability, seed)
    if method == 1:
        G = nx.gnp_random_graph(node, probability, seed, False)
        G = G.to_undirected()
    if method == 2:
        G = nx.gnp_random_graph(node, probability, seed, False)
        G = G.to_undirected()
    return G

def generate_regular_graph(d,n):
    if d*n % 2 == 1:
        n = n+1
    G = random_regular_graph(d, n, seed=None)

    return G

def generate_k_clique_graph(n,k):

    size_clique = 0
    G = 0

    while size_clique != n:
        random.seed()
        node = random.randint(0, 100)
        seed = random.randint(0, 100)
        probability = random.uniform(0, 1)

        #generate first Graph
        G = generate_graph(node, probability, seed, 1)
        size_clique = 0
        for clq in nx.clique.find_cliques(G):
            if len(clq) >= k:
                size_clique = size_clique + 1

    nx.set_edge_attributes(G, {e: {'weight': random.randint(1, 10)} for e in G.edges})
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(0, 10)
    return G



def get_graph_Data(G,node):
    edge_labels = dict(((u, v), d["weight"]) for u, v, d in G.edges(data=True))
    weights_vor_Array = list(edge_labels.items())
    weights_array = np.array(weights_vor_Array)
    weights_array_node = weights_array[:, 0]
    liste = weights_array_node.tolist()
    df = pd.DataFrame(liste)
    list_neu = df[0].tolist()
    list_neu_1 = df[1].tolist()
    backagain = np.array(list_neu)
    backagain_1 = np.array(list_neu_1)
    weights_array_weights = weights_array[:, 1]
    gewichte = np.concatenate((backagain, weights_array_weights), axis=0)
    table_weights = Table([backagain, weights_array_weights], names=('Knoten', 'Gewichte'))
    return [list_neu, list_neu_1, backagain, backagain_1, weights_array_weights, gewichte, table_weights, edge_labels, weights_array_node]

#calculating degree-centality
def calculate_degree_centrality(G,node):
    #getting graph data
    data = get_graph_Data(G, node)
    backagain = data[2]
    weights_array_weights = data[4]

    weights = np.zeros(node)
    for i in range(0, len(backagain)):
        a = weights_array_weights[i]
        weights[backagain[i]] = weights[backagain[i]] + a

    #normalizing weights
    weights = weights/10
    #applying formula
    centrality = weights/(node-1)
    print('Graph', G)
    print('sum is', weights)
    print('weighted degree centrality is', centrality)
    return centrality




#calculating closness-centality
def calcuate_closness_centrality(G,node):
    #calculating Adjacent-matrix
    dim = node
    data = get_graph_Data(G,node)
    list_neu = data[0]
    list_neu_1 = data[1]
    weights_array_weights = data[4]

    Adja = np.zeros((dim,dim))
    for i in range(0,len(list_neu)):
        a = list_neu[i]
        b = list_neu_1[i]
        w = weights_array_weights[i]
        Adja[a][b] = w
        Adja[b][a] = w

    #print("Adjacent-matrix: ", Adja)
    #apply floyd warshall for shortest path
    A = floyd_warshall(Adja)
    #print("The shortest Path to each point is represnted here: " , A)
    #print("The shortest Path without inf: " , A)
    #calculating the centrality for each point
    closness_centrality = np.zeros(node)
    for i in range(0,node):
        closness_centrality[i] = (node - 1)/sum(A[i,:])

    return closness_centrality

#calculating between-centality
def calculate_bettween_centrality(G,node):

    dim = node
    data = get_graph_Data(G, node)
    list_neu = data[0]
    list_neu_1 = data[1]
    weights_array_weights = data[4]

    Adja = np.zeros((dim, dim))
    for i in range(0, len(list_neu)):
        a = list_neu[i]
        b = list_neu_1[i]
        w = weights_array_weights[i]
        Adja[a][b] = w
        Adja[b][a] = w
    predA = find_shortest_path(Adja,node)
    #now we have a matrix where the shortest paths are generated
    #getting all the paths
    #todo: adding up the numbers in the predA. Somehow find out how many shortest path exist

def calculate_eigenvector_centrality(G,node):
    dim = node
    data = get_graph_Data(G,node)
    list_neu = data[0]
    list_neu_1 = data[1]
    weights_array_weights = data[4]

    Adja = np.zeros((dim, dim))
    for i in range(0, len(list_neu)):
        a = list_neu[i]
        b = list_neu_1[i]
        w = weights_array_weights[i]
        Adja[a][b] = w #instead maby use 1
        Adja[b][a] = w #instead maby use 1

    #Eigenvalues, Eigenvectors = np.linalg.eig(np.transpose(Adja))
    #max_eigen = max(Eigenvalues)
    #arg_max_eigen = np.argmax(Eigenvalues)
    #max_vektor = Eigenvectors[arg_max_eigen]

    #print(Eigenvalues)
    #print(Eigenvectors)
    #print("the max vector is: ",max_vektor)


    #e = np.ones(node)
    #A_help = np.diag(Adja@e)
    #N = np.linalg.inv(A_help)@Adja
    #Eigenvalues, Eigenvectors = np.linalg.eig(np.transpose(N))
    #max_eigen = max(Eigenvalues)
    #arg_max_eigen = np.argmax(Eigenvalues)
    #max_vektor = Eigenvectors[arg_max_eigen]

    #return max_vektor






#dijkstra help
def find_shortest_path(G, node):
    print(G)
    dim = node
    predMatrix = np.zeros((dim, dim))
    Adja = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i != j and G[i][j] == 0:
                Adja[i][j] = np.inf
            else:
                Adja[i][j] = G[i][j]
    #paths = np.zeros((dim, dim))

    for i in range(0, node):
        for j in range(0, node):
            #paths[i][j] = [[i]]
            predMatrix[i][j] = 0
            if not i == j or Adja[i][j] == np.inf:
                predMatrix[i][j] = i

    for k in range(0, node):
        for i in range(0, node):
            if Adja[i][k] < np.inf:
                for j in range(0, node):
                    if Adja[k][j] < np.inf:
                        if Adja[i][k] + Adja[k][j] < Adja[i][j]:
                            Adja[i][j] = Adja[i][k] + Adja[k][j]
                            predMatrix[i][j] = predMatrix[k][j]

    #the shortest way to the point from the point, is not possible
    for i in range(dim):
        predMatrix[i][i] = np.inf

    #prints can be removed
    print("predMatrix:")
    print(predMatrix)
    print("Floyd-Warshall:")
    print(floyd_warshall(G))

    return predMatrix



#calculating eigenvektor-centality

#create graph with given number of cliques
def k_cliques_graph(G):
    i = 0
    for clq in nx.clique.find_cliques(G):
        i = i + 1
        print(clq)

    return i

def generate_table(G, node):
    data = get_graph_Data(G, node)
    nodes = np.arange(node)
    degree = calculate_degree_centrality(G, node)
    close = calcuate_closness_centrality(G, node)
   # between = calculate_bettween_centrality(G, node)
   # eigen = calculate_eigenvector_centrality(G, node)
    clique = k_cliques_graph(G)

    t = Table([nodes, degree, close], names=('Nodes', 'weighted Degree', 'closeness'))
    print(t)
    table = tabulate(t, tablefmt="latex")
    #print(table)
    return table

#draw the Graph
def draw_Graph(G, G2, node):
    plt.figure(figsize=(14, 7))
    plt.subplot(211)
    plt.title('Random Graph G')
    pos = nx.random_layout(G)
    nx.draw(G, pos, edge_color='grey', width=0.7, linewidths=0.1, node_color='lightblue', node_size=250, labels={node: node for node in G.nodes()}, font_size = 9)
    data = get_graph_Data(G, node)
    edge_labels = data[7]
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkblue')
    plt.subplot(212)
    plt.title('Random Graph G2')
    pos_G2 = nx.random_layout(G2)
    nx.draw(G2, pos_G2, edge_color='grey', width=0.7, linewidths=0.1, node_color='lightpink', node_size=250, labels={node: node for node in G2.nodes()}, font_size = 9)
    data_G2 = get_graph_Data(G2, node)
    edge_labels_G2 = data_G2[7]
    nx.draw_networkx_edge_labels(G2, pos_G2, edge_labels=edge_labels_G2, font_size=8, font_color='darkblue')
    plt.show()






##some tests
random.seed()
node = random.randint(15, 20)
print('node has value', node)
seed = random.randint(1, 10)
print('seed has value', seed)
probability = random.uniform(0.2, 0.5)
print(probability)


#G = generate_graph(node, probability, seed, 1)
G = generate_k_clique_graph(5, 0)
#nx.set_edge_attributes(G, {e: {'weight': random.randint(1, 10)} for e in G.edges})
#for (u, v) in G.edges():
    #G.edges[u, v]['weight'] = random.randint(0, 10)

G2 = generate_k_clique_graph(5, 0)
#print the tests
def print_all(G, G2, node):
    print("degree_centrality G")
    print(calculate_degree_centrality(G, node))
    print("closness_centrality G")
    print(calcuate_closness_centrality(G, node))
    print("between_centrality G")
    print(calculate_bettween_centrality(G, node))
    print("eigenvector_centrality G")
    print(calculate_eigenvector_centrality(G, node))
    print("cliques G")
    print(calculate_eigenvector_centrality(G, len(G.nodes)))
    print(G)
    print("amount of cliques G")
    print(k_cliques_graph(G))
    print("Latex Tabelle G")
    print(generate_table(G, node))
    print("degree_centrality G2")
    print(calculate_degree_centrality(G2, node))
    print("closness_centrality G2")
    print(calcuate_closness_centrality(G2, node))
    print("between_centrality G2")
    print(calculate_bettween_centrality(G2, node))
    print("eigenvector_centrality G2")
    print(calculate_eigenvector_centrality(G2, node))
    print("cliques G2")
    print(calculate_eigenvector_centrality(G2, len(G2.nodes)))
    print(G2)
    print("amount of cliques G2")
    print(k_cliques_graph(G2))
    print("Latex Tabelle G2")
    print(generate_table(G2, node))
    print("Latex Tabelle G2")
    print(generate_table(G2, node))
    draw_Graph(G, G2, node)
    #draw_Graph(G,node)

print_all(G, G2, node)


