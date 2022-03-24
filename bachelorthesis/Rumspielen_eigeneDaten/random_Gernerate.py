#matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from random import random


def ER(n, p):
    V = set([v for v in range(n)])
    E = set()
    for combination in combinations(V, 2):
        a = random()
        if a < p:
            E.add(combination)

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g


n = 30
p = 0.1
G1 = ER(n, p)
pos = nx.spring_layout(G1)
nx.draw_networkx(G1)
c1=nx.closeness_centrality(G1)
print(c1)
G2=nx.erdos_renyi_graph(100,0.6)
c2=nx.closeness_centrality(G2)
print(c2)
plt.show()
