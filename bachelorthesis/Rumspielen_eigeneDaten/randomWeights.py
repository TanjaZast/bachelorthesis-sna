import networkx as nx
import nxviz as nv
import pandas as pd
import matplotlib.pyplot as plt

# read all the 5 csv files
# keep only the distinct pairs of source target since we will ignore the books and the weights

GOTbooks = ["book1.csv", "book2.csv", "book3.csv", "book4.csv", "book5.csv"]

list = []

for i in GOTbooks:
    tmp = pd.read_csv(i)
    list.append(tmp)

df = pd.concat(list, axis=0, ignore_index=True)

df = df[['Source', 'Target']]
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)


# create the networkx object

G = nx.from_pandas_edgelist(df,  source='Source', target='Target')

# How to get the number of nodes

print(len(G.nodes()))

# How to get the number of edges

print(len(G.edges()))
