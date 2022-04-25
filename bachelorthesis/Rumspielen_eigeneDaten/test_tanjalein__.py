import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout

G = nx.Graph()
G.add_edges_from(
[(3, 1), (3, 4), (3, 6), (3, 5), (4, 5), (4, 6), (6, 5), (1,2), (2,7), (7,8), (8,1), (7,1), (8, 2), (3,9), (9,10), (10,11), (9, 11)])
nx.draw_networkx(G, node_size = 300, font_size=10, node_color= 'lightblue')
plt.show()
