import random

import numpy as np
from networkx import eigenvector_centrality
from networkx.generators.random_graphs import erdos_renyi_graph, dense_gnm_random_graph, newman_watts_strogatz_graph, \
    random_regular_graph, barabasi_albert_graph, powerlaw_cluster_graph
import networkx as nx
import matplotlib.pyplot as plt
from numpy import transpose
from astropy.table import Table
from tabulate import tabulate
i = 0
j = 0
l = 0
q = 0
r = 0
s = 0
#n = random.randint(30, 100)
n = 20
print(n)
m = int(0.1*(n*(n-1)/2))
print(m)
k = random.randint(10, 20)
print(k)
d = random.randint(1, 10)
print(d)
p = 0.1
G1 = erdos_renyi_graph(n, p)
G2 = dense_gnm_random_graph(n, m)
G3 = newman_watts_strogatz_graph(n, k, p)
G4 = random_regular_graph(d, n)
m = n -4
G5 = barabasi_albert_graph(n, m)
G6 = powerlaw_cluster_graph(n, m, p)
deg_centrality1 = nx.degree_centrality(G1)
deg_centrality2 = nx.degree_centrality(G2)
#in_deg_centrality = nx.in_degree_centrality(G)
#out_deg_centrality = nx.out_degree_centrality(G)

close_centrality1 = nx.closeness_centrality(G1)
bet_centrality1 = nx.betweenness_centrality(G1, normalized = True, endpoints = False)
eigenvector1 = eigenvector_centrality(G1, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq1 in nx.clique.find_cliques(G1):
    i = i +1
    print (clq1)
print(i)
print(G1.nodes)
print(G1.edges)

G1a0 = np.arange(n)
#print(G1a0)
G1dict1 = deg_centrality1
#print(G1dict1)
G1data1 = list(G1dict1.items())
G1an_array1 = np.array(G1data1)
G1a1 = transpose(G1an_array1)
G1a1 = G1a1[1]
#print(G1a1)

G1dict2 = close_centrality1
#print(G1dict2)
G1data2 = list(G1dict2.items())
G1an_array2 = np.array(G1data2)
G1a2 = transpose(G1an_array2)
G1a2 = G1a2[1]
#print(G1a2)

G1dict3 = bet_centrality1
#print(G1dict3)
G1data3 = list(G1dict3.items())
G1an_array3 = np.array(G1data3)
G1a3 = transpose(G1an_array3)
G1a3 = G1a3[1]
#print(G1a3)

G1dict4 = eigenvector1
#print(G1dict4)
G1data4 = list(G1dict4.items())
G1an_array4 = np.array(G1data4)
G1a4 = transpose(G1an_array4)
G1a4 = G1a4[1]
#print(G1a4)

G1a = np.concatenate((G1a0, G1a1, G1a2, G1a3, G1a4), axis=0)
#print(G1a)

G1z1 = G1a0
G1z2 = G1a1
G1z3 = G1a2
G1z4 = G1a3
G1z5 = G1a4
t1 = Table([G1z1, G1z2, G1z3, G1z4, G1z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t1)
print(tabulate(t1, tablefmt="latex"))

close_centrality2 = nx.closeness_centrality(G2)
bet_centrality2 = nx.betweenness_centrality(G2, normalized = True, endpoints = False)
eigenvector2 = eigenvector_centrality(G2, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq2 in nx.clique.find_cliques(G2):
    j = j +1
    print (clq2)
print(j)
print(G2.nodes)
print(G2.edges)
G2a0 = np.arange(n)
#print(G2a0)
G2dict1 = deg_centrality2
#print(G2dict1)
G2data1 = list(G2dict1.items())
G2an_array1 = np.array(G2data1)
G2a1 = transpose(G2an_array1)
G2a1 = G2a1[1]
#print(G2a1)

G2dict2 = close_centrality2
#print(G2dict2)
G2data2 = list(G2dict2.items())
G2an_array2 = np.array(G2data2)
G2a2 = transpose(G2an_array2)
G2a2 = G2a2[1]
#print(G2a2)

G2dict3 = bet_centrality2
#print(G2dict3)
G2data3 = list(G2dict3.items())
G2an_array3 = np.array(G2data3)
G2a3 = transpose(G2an_array3)
G2a3 = G2a3[1]
#print(G2a3)

G2dict4 = eigenvector2
#print(G2dict4)
G2data4 = list(G2dict4.items())
G2an_array4 = np.array(G2data4)
G2a4 = transpose(G2an_array4)
G2a4 = G2a4[1]
#print(G2a4)

G2a = np.concatenate((G2a0, G2a1, G2a2, G2a3, G2a4), axis=0)
#print(G2a)

G2z1 = G2a0
G2z2 = G2a1
G2z3 = G2a2
G2z4 = G2a3
G2z5 = G2a4
t2 = Table([G2z1, G2z2, G2z3, G2z4, G2z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t2)
print(tabulate(t2, tablefmt="latex"))

deg_centrality3 = nx.degree_centrality(G3)

close_centrality3 = nx.closeness_centrality(G3)
bet_centrality3 = nx.betweenness_centrality(G3, normalized = True, endpoints = False)
eigenvector3 = eigenvector_centrality(G3, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq3 in nx.clique.find_cliques(G3):
    l = l +1
    print (clq3)
print(l)
print(G3.nodes)
print(G3.edges)
G3a0 = np.arange(n)
#print(G3a0)
G3dict1 = deg_centrality3
#print(G3dict1)
G3data1 = list(G3dict1.items())
G3an_array1 = np.array(G3data1)
G3a1 = transpose(G3an_array1)
G3a1 = G3a1[1]
#print(G3a1)

G3dict2 = close_centrality3
#print(G3dict2)
G3data2 = list(G3dict2.items())
G3an_array2 = np.array(G3data2)
G3a2 = transpose(G3an_array2)
G3a2 = G3a2[1]
#print(G3a2)

G3dict3 = bet_centrality3
#print(G3dict3)
G3data3 = list(G3dict3.items())
G3an_array3 = np.array(G3data3)
G3a3 = transpose(G3an_array3)
G3a3 = G3a3[1]
#print(G3a3)

G3dict4 = eigenvector3
#print(G3dict4)
G3data4 = list(G3dict4.items())
G3an_array4 = np.array(G3data4)
G3a4 = transpose(G3an_array4)
G3a4 = G3a4[1]
#print(G3a4)

G3a = np.concatenate((G3a0, G3a1, G3a2, G3a3, G3a4), axis=0)
#print(G3a)

G3z1 = G3a0
G3z2 = G3a1
G3z3 = G3a2
G3z4 = G3a3
G3z5 = G3a4
t3 = Table([G3z1, G3z2, G3z3, G3z4, G3z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t3)
print(tabulate(t3, tablefmt="latex"))

deg_centrality4 = nx.degree_centrality(G4)

close_centrality4 = nx.closeness_centrality(G4)
bet_centrality4 = nx.betweenness_centrality(G4, normalized = True, endpoints = False)
eigenvector4 = eigenvector_centrality(G4, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq4 in nx.clique.find_cliques(G4):
    q = q +1
    print (clq4)
print(q)
print(G4.nodes)
print(G4.edges)
print(deg_centrality4)
G4a0 = np.arange(n)
#print(G4a0)
G4dict1 = deg_centrality4
#print(G4dict1)
G4data1 = list(G4dict1.items())
G4an_array1 = np.array(G4data1)
G4a1 = transpose(G4an_array1)
G4a1 = G4a1[1]
#print(G4a1)

G4dict2 = close_centrality4
#print(G4dict2)
G4data2 = list(G4dict2.items())
G4an_array2 = np.array(G4data2)
G4a2 = transpose(G4an_array2)
G4a2 = G4a2[1]
#print(G4a2)

G4dict3 = bet_centrality4
#print(G4dict3)
G4data3 = list(G4dict3.items())
G4an_array3 = np.array(G4data3)
G4a3 = transpose(G4an_array3)
G4a3 = G4a3[1]
#print(G4a3)

G4dict4 = eigenvector4
#print(G4dict4)
G4data4 = list(G4dict4.items())
G4an_array4 = np.array(G4data4)
G4a4 = transpose(G4an_array4)
G4a4 = G4a4[1]
#print(G4a4)

G4a = np.concatenate((G4a0, G4a1, G4a2, G4a3, G4a4), axis=0)
#print(G4a)

G4z1 = G4a0
G4z2 = G4a1
G4z3 = G4a2
G4z4 = G4a3
G4z5 = G4a4
t4 = Table([G4z1, G4z2, G4z3, G4z4, G4z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t4)
print(tabulate(t4, tablefmt="latex"))

deg_centrality5 = nx.degree_centrality(G5)

close_centrality5 = nx.closeness_centrality(G5)
bet_centrality5 = nx.betweenness_centrality(G5, normalized = True, endpoints = False)
eigenvector5 = eigenvector_centrality(G5, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq5 in nx.clique.find_cliques(G5):
    r = r +1
    print (clq5)
print(r)
print(G5.nodes)
print(G5.edges)
print(deg_centrality5)
G5a0 = np.arange(n)
#print(G5a0)
G5dict1 = deg_centrality5
#print(G5dict1)
G5data1 = list(G5dict1.items())
G5an_array1 = np.array(G5data1)
G5a1 = transpose(G5an_array1)
G5a1 = G5a1[1]
#print(G5a1)

G5dict2 = close_centrality5
#print(G5dict2)
G5data2 = list(G5dict2.items())
G5an_array2 = np.array(G5data2)
G5a2 = transpose(G5an_array2)
G5a2 = G5a2[1]
#print(G5a2)

G5dict3 = bet_centrality5
#print(G5dict3)
G5data3 = list(G5dict3.items())
G5an_array3 = np.array(G5data3)
G5a3 = transpose(G5an_array3)
G5a3 = G5a3[1]
#print(G5a3)

G5dict4 = eigenvector5
#print(G5dict4)
G5data4 = list(G5dict4.items())
G5an_array4 = np.array(G5data4)
G5a4 = transpose(G5an_array4)
G5a4 = G5a4[1]
#print(G5a4)

G5a = np.concatenate((G5a0, G5a1, G5a2, G5a3, G5a4), axis=0)
#print(G5a)

G5z1 = G5a0
G5z2 = G5a1
G5z3 = G5a2
G5z4 = G5a3
G5z5 = G5a4
t5 = Table([G5z1, G5z2, G5z3, G5z4, G5z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t5)
print(tabulate(t5, tablefmt="latex"))

deg_centrality6 = nx.degree_centrality(G6)

close_centrality6 = nx.closeness_centrality(G6)
bet_centrality6 = nx.betweenness_centrality(G6, normalized = True, endpoints = False)
eigenvector6 = eigenvector_centrality(G6, max_iter=100, tol=1e-06, nstart=None, weight=None)
for clq6 in nx.clique.find_cliques(G6):
    s = s +1
    print (clq6)
print(s)
print(G6.nodes)
print(G6.edges)
print(deg_centrality6)
G6a0 = np.arange(n)
#print(G6a0)
G6dict1 = deg_centrality6
#print(G6dict1)
G6data1 = list(G6dict1.items())
G6an_array1 = np.array(G6data1)
G6a1 = transpose(G6an_array1)
G6a1 = G6a1[1]
#print(G6a1)

G6dict2 = close_centrality6
#print(G6dict2)
G6data2 = list(G6dict2.items())
G6an_array2 = np.array(G6data2)
G6a2 = transpose(G6an_array2)
G6a2 = G6a2[1]
#print(G6a2)

G6dict3 = bet_centrality6
#print(G6dict3)
G6data3 = list(G6dict3.items())
G6an_array3 = np.array(G6data3)
G6a3 = transpose(G6an_array3)
G6a3 = G6a3[1]
#print(G6a3)

G6dict4 = eigenvector6
#print(G6dict4)
G6data4 = list(G6dict4.items())
G6an_array4 = np.array(G6data4)
G6a4 = transpose(G6an_array4)
G6a4 = G6a4[1]
#print(G6a4)

G6a = np.concatenate((G6a0, G6a1, G6a2, G6a3, G6a4), axis=0)
#print(G6a)

G6z1 = G6a0
G6z2 = G6a1
G6z3 = G6a2
G6z4 = G6a3
G6z5 = G6a4
t6 = Table([G6z1, G6z2, G6z3, G6z4, G6z5], names=('Nodes', 'Degree', 'closeness', 'between', 'eigenvektor'))
print(t6)
print(tabulate(t6, tablefmt="latex"))

plt.figure(figsize=(14, 7))
plt.subplot(321)
plt.title('Erdos Renyi')
nx.draw_networkx(G1, with_labels = True, node_color='lightblue')
plt.subplot(322)
plt.title('dense_gnm_random_graph')
nx.draw_networkx(G2, with_labels = True, node_color='green')
plt.subplot(323)
plt.title('newman_watts_strogatz_graph')
nx.draw_networkx(G3, with_labels = True, node_color='yellow')
plt.subplot(324)
plt.title('random_regular_graph')
nx.draw_networkx(G4, with_labels = True, node_color='purple')
plt.subplot(325)
plt.title('barabasi_albert_graph')
nx.draw_networkx(G5, with_labels = True, node_color='brown')
plt.subplot(326)
plt.title('powerlaw_cluster_graph')
nx.draw_networkx(G6, with_labels = True, node_color='blue')
plt.show()


