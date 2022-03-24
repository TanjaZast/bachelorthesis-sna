import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.pyplot import figure


df = pd.read_csv('/Users/tanjazast/Desktop/Bachelorarbeit/CSV/combinedFriendProfile.csv')
# pick only important weights (hard threshold)
print(df)
df= pd.read_csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/combinedFriendProfile.csv", sep=r';')
print(df)
df1 = df[['timestamp', 'data.name']]
df2 = df[['timestamp.name', 'name']]
print(df1)
print(df2)


G1 = nx.erdos_renyi_graph(20, p=0.1)
G2 = nx.erdos_renyi_graph(20, p=0.1)
G1 = nx.from_pandas_edgelist(df1, 'timestamp', 'data.name')
G2 = nx.from_pandas_edgelist(df2, 'timestamp.name', 'name')


figure(figsize=(10, 8))
plt.subplot(211)
nx.draw_networkx(G1, with_labels = True, node_color='blue')
plt.subplot(212)
nx.draw_networkx(G2, with_labels = True, node_color='green')
plt.show()
