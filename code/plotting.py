import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

"""
Plotting code used for general visualization
"""

# tie_decay_result = np.load("tie_decay_1e3.npy").item()

print ("Now start plotting")

G = nx.Graph()
tie_decay_edges = [(k[0], k[1], v) for k, v in tie_decay_result.items()
                  if v >= 10 ** -100]
G.add_weighted_edges_from(tie_decay_edges)

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
colors = ['r' if w > 10 ** -10 else 'b' for w in weights]
width = [1 if w > 10 ** -10 else 0.1 for w in weights]
pos = nx.spring_layout(G)

plt.figure()
nx.draw(G, pos, node_size=5, node_color='b', edgelist=edges, edge_color=colors, width=width)
plt.savefig("tie_decay_example.png")
plt.close()
