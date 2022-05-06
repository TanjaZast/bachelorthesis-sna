import random
from math import sqrt
import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators import community

from numpy import transpose
from astropy.table import Table
from tabulate import tabulate
import numpy as np
import seaborn as sns



# generate random matrices

def random_adjacency_matrix(n):
    # matrix = np.random.uniform(0.0, 0.99, size=(n, n))
    # Vorteil: ca gleich viele Knoten um 0.5
    matrix = np.random.uniform(0.5, 0.99, size=(n, n))

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 1
    matrix2 = matrix

    for i in range(n):

        # print("die Wkeit für die Existenz einer Verbindung lautet:", prob)
        for k in range(0, n):
            prob = random.uniform(0.2, 1)
            if matrix2[i][k] > prob:
                matrix2[i][k] = 0
            else:
                matrix2[i][k] = 1

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix2[j][i] = matrix2[i][j]

    return matrix2


# print('Die Matrix sieht wie folgt aus', random_adjacency_matrix(20))

def tables(degree, closeness, between):
    Ga0 = degree
    Gdata0 = list(Ga0.items())
    Gan_array0 = np.array(Gdata0)
    Ga0 = transpose(Gan_array0)
    Ga0 = Ga0[0]

    Gdict = degree
    Gdata = list(Gdict.items())
    Gan_array = np.array(Gdata)
    Ga1 = transpose(Gan_array)
    Ga1 = Ga1[1]

    Gdict2 = closeness
    Gdata2 = list(Gdict2.items())
    Gan_array2 = np.array(Gdata2)
    Ga2 = transpose(Gan_array2)
    Ga2 = Ga2[1]
    print('closeness centrality looks like', Ga2)

    Gdict3 = between
    Gdata3 = list(Gdict3.items())
    Gan_array3 = np.array(Gdata3)
    Ga3 = transpose(Gan_array3)
    Ga3 = Ga3[1]

    #Ga = np.concatenate((Ga0, Ga1, Ga2, Ga3), axis=0)
    #print('Step before converting into table:', Ga)

    Gz1 = Ga0
    Gz2 = Ga1
    Gz3 = Ga2
    Gz4 = Ga3
    t = Table([Gz1, Gz2, Gz3, Gz4], names=('Nodes', 'Degree', 'closeness', 'between'))
    #print(t)
    print(tabulate(t, tablefmt="latex"))

def generate_denisity(bars, centralities):

    centralities = np.array(centralities)

    #initializing distribution
    min =  np.min(centralities)
    max =  np.max(centralities)
    dist = np.zeros(bars)
    intv = max - min
    x = np.linspace(np.min(centralities),np.max(centralities),bars)
    barsize = ((max - min)/bars)*0.8

    for d in centralities:
        dist[int(((d-min)*bars)/intv)-1] = dist[int(((d-min)*bars)/intv)-1] + 1

    return [dist, x, barsize]



# generating and showing graph
def show_graph_with_labels(adjacency_matrices):
    # generate big graph with all individual matrices combined
    global clq, lenght
    big_graph = unite_graphs(adjacency_matrices)
    #print(big_graph.shape)
    bars = 100

    # generate edges for big graph
    rows_big, cols_big = np.where(big_graph == 1)
    edges_big = zip(rows_big.tolist(), cols_big.tolist())
    gr_big = nx.Graph()
    gr_big.add_edges_from(edges_big)




    fig1, axes = plt.subplots(2,2)
    fig1.set_size_inches(9,7)
    #mu, std = norm.fit(centralities)
    #print("mu is", mu,"std is", std)

    #p = norm.pdf(x, mu, std)
    #axes.plot(x, p, color="darkblue", linewidth=1.5)
    #print("the distribution looks like  ", x_values)
    #degree_centralities
    degree = calculate_degree_centrality(gr_big)
    print("degree looks like", degree)
    sns.distplot(degree, ax = axes[0][0], bins = bars, label= 'distribution')
    sns.rugplot(degree, clip_on = False,alpha = 0.01, height = -0.02, ax = axes[0][0])
    axes[0][0].set_title("distribution over degree-centrality")
    axes[0][0].legend()

    #closness_centralities
    closeness = calculate_closness_centrality(gr_big)
    sns.distplot(closeness, ax=axes[0][1], bins=bars, label='distribution')
    sns.rugplot(closeness, clip_on=False, alpha=0.01, height=-0.02, ax=axes[0][1])
    axes[0][1].set_title("distribution over closeness-centrality")
    axes[0][1].legend()

    #between_centralities
    between = calculate_between_centrality(gr_big)
    value = np.log(between)
    sns.distplot(value, ax=axes[1][0], bins=bars, label='distribution')
    sns.rugplot(value, clip_on=False, alpha=0.01, height=-0.02, ax=axes[1][0])
    axes[1][0].set_title("distribution over betweenness-centrality")
    axes[1][0].legend()

    #distribution n times degree_centralities
    eigenvector = calculate_eigenvector_centrality(gr_big)
    sns.distplot(np.log(eigenvector), ax=axes[1][1], bins=bars, label='distribution')
    sns.rugplot(np.log(eigenvector), clip_on=False, alpha=0.01, height=-0.02, ax=axes[1][1])
    axes[1][1].set_title("distribution over eigenvector-centrality")
    axes[1][1].legend()
    plt.savefig("/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/Plots/generatedPlotDensity.png")
    plt.show()

    d = nx.degree_centrality(gr_big)
    c = nx.closeness_centrality(gr_big)
    b = nx.betweenness_centrality(gr_big)
    #a = nx.eigenvector_centrality(gr_big)
    tables(d, c, b)


    i = 0
    lenght = []
    for clq in nx.clique.find_cliques(gr_big):
        i = i + 1
        lenght.append(len(clq))
    print('amount of cliques', i)
    print('biggest clique', max(lenght))
    node = int(sqrt(big_graph.size))
    print('number of nodes', node)

    pos = nx.spring_layout(gr_big, seed=196900)
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(7)
    edge = gr_big.number_of_edges()
    print('number of edges', edge)
    gr_big.number_of_nodes()
    nx.draw(gr_big, pos, node_color=range(len(gr_big)), font_size=2, cmap=plt.cm.tab10,
            node_size=15, edge_color="#D4D5CE", width=0.4, linewidths=0.4)
    plt.savefig(
        "/Users/tanjazast/Desktop/Bachelorthesis/bachelorthesis-sna/bachelorthesis/Plots/generatedPlot.png")
    plt.show()


# function to unite list of graphs
def unite_graphs(graphs):
    if len(graphs) == 1:
        return graphs[0]

    dim = 0
    for g in graphs:
        dim = dim + len(g)
    big_graph = np.zeros((dim, dim))
    graph_lengths = np.zeros(len(graphs))

    # first get the dimensions correctly
    for i in range(0, len(graphs)):
        graph_lengths[i] = len(graphs[i])

    # get edges right some positions with int because variable l is weird
    for i in range(0, len(graphs)):

        a = random.randint(0, len(graphs))
        b = random.randint(0, len(graphs))

        for j in range(0, len(graphs[i])):
            for k in range(0, len(graphs[i])):
                graph = graphs[i]

                l = sum(graph_lengths[:i])
                big_graph[int(l + j)][int(l + k)] = graph[j][k]
                big_graph[int(l + k)][int(l + j)] = graph[k][j]
                big_graph[int(l + a)][int(l + graph_lengths[i] + b) % dim] = 1
                big_graph[int(l + graph_lengths[i] + b) % dim][int(l + a)] = 1

    ###make sure that random connections between points exist.
    p = 0.0001 #probability, that there is a connection between node i and j
    for i in range(len(big_graph)):
        for j in range(len(big_graph[0])):
            r = random.uniform(0,1)
            if r < p:
                big_graph[i][j] = 1

    ###kann ausgelagert werden
    return big_graph


# find node with maximum edges (max count of values = 1)
def find_max_node(graph):
    counterZwei = 0
    a = 0
    for i in range(0, len(graph)):
        counterEins = 0
        for j in range(0, len(graph)):
            if graph[i][j] != 0:
                # counterEins = counterEins + 1
                counterEins = counterEins + 1
            if counterEins > counterZwei:
                counterEins = counterZwei
                a = i
    return a


def maximum(matrix):
    max_value = None
    for num in matrix:
        if max_value is None or num > max_value:
            max_value = num
    return max_value


def random_matrix_generator(n, p):
    return random_adjacency_matrix(n, p)


def graph_appender(n):
    graphs = []
    for i in range(n):
        # groups have the size between n and 2n
        k = random.randint(100, 300)
        pr = random.uniform(0.5, 1)
        graphs.append(random_matrix_generator(k, pr))

    # print('List of Graphs is', graphs)
    return graphs


#####################
# prob: defines the probability that there is a connection from point i to j and they are not in the same subgraph
# dense: defines the probability, that there is a connection between point j and i and they are in teh same subgraph
def cluster_generator(n, prob, dense):
    adja = np.zeros((n, n))
    c = n
    cluster = []
    while c != 0:
        c_1 = random.randint(4, int(n / 2))
        if c_1 > c:
            c_1 = c
        c = c - c_1
        cluster.append(c_1)

    # genrate adjazent-matrix
    for i in range(0, len(cluster) - 1):
        for j in range(0, n):
            for k in range(0, n):
                if sum(cluster[:i + 1]) >= j >= sum(cluster[:i]) and sum(cluster[:i + 1]) > k >= sum(
                        cluster[:i]):
                    rand = random.uniform(0, 1)
                    if rand <= dense:
                        adja[j][k] = 1
                        adja[k][j] = 1
                if sum(cluster[:i + 1]) >= j >= sum(cluster[:i]) and (
                        k > sum(cluster[:i + 1]) or k < sum(cluster[:i])):
                    rand = random.uniform(0, 1)
                    if rand <= prob:
                        adja[j][k] = 1
                        adja[k][j] = 1
    # clear diagonals
    for i in range(n):
        adja[i][i] = 0

    # print('adja looks like', adja)
    return adja

# help method to calculate centrality with adjacent matrix
def calculate_degree_centrality(G):
    #return [sum(row) / len(adja) for row in adja]
    degree = nx.degree_centrality(G)
    listDegree = list(degree.items())
    arrayDegree = np.array(listDegree)
    transposed = transpose(arrayDegree)
    degree_centrality = transposed[1]
    #print('closeness centrality looks like', close_centrality)
    return degree_centrality

def calculate_closness_centrality(G):

    closeness = nx.closeness_centrality(G)
    listCloseness = list(closeness.items())
    arrayCloseness = np.array(listCloseness)
    transposed = transpose(arrayCloseness)
    close_centrality = transposed[1]
    #print('closeness centrality looks like', close_centrality)
    return close_centrality

def calculate_between_centrality(G):

    between = nx.betweenness_centrality(G)
    listBetween = list(between.items())
    arrayBetween = np.array(listBetween)
    transposed = transpose(arrayBetween)
    between_centrality = transposed[1]
    #print('betweenness centrality looks like', between_centrality)
    return between_centrality

def calculate_eigenvector_centrality(G):

    eigenvector = nx.eigenvector_centrality_numpy(G)
    listEigenvector = list(eigenvector.items())
    arrayEigenvector = np.array(listEigenvector)
    transposed = transpose(arrayEigenvector)
    eigenvector_centrality = transposed[1]
   #print('closeness centrality looks like', eigenvector_centrality)
    return eigenvector_centrality


#p = random.uniform(0, 1)
graphs = graph_appender(7)
#n = random.randint(2000, 2500)
#amount = random.randint(2000, 2500)
show_graph_with_labels(graphs)

