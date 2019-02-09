# import os
import pickle
import numpy as np
# import networkx as nx

# os.chdir("/Users/qinyichen/Documents/brexit-tie-decay/Data/data_3_9jun")
# data = pd.read_csv("edge_list_updated.csv")

# source  = np.array(data["source_id"])
# target  = np.array(data["target_id"])
# N = max(max(source), max(target))      # N: number of elements

# G = nx.from_pandas_edgelist(data, "source_id", "target_id", "Weight")
# A = nx.adjacency_matrix(G)

with open('edge_dict.pkl','rb') as f:
    data = pickle.load(f)

def tie_decay_matrix(edge_list, T, t_start=0, dt=60, alpha=0.1):

    """
    INPUT:
    (1) edge_list: a dictionary with keys being (i, j) and values being
                   the list of (time, weight) tuple
    (2) T: end of time of interest
    (3) t_start: start of time of interest
    (4) dt: size of time step
    (5) alpha: decay coefficient

    OUTPUT:
    B(T): connection matrix at time T
    """
    # Initialize B at t = 0 with no activities
    B = {}

    # Update B at different time
    for t in range(t_start, T, dt):
        print ("start calculating B at t =", t)

        # entries of B decay from t to t+dt
        B.update((k, v*np.exp(-alpha * dt)) for k, v in B.items())

        # Add the new interactions
        for edge, attrs in edge_list.items():
            # print ("now checking:", edge, attrs)
            # check whether there are iterations in this time step
            while attrs and attrs[0][0] >= t and attrs[0][0] < t+dt:
                time, weight = attrs.pop()
                B[edge] = B.get(edge, 0) + weight

        # Update the remaining edge list; not sure if necessary
        edge_list = {k:edge_list[k] for k in edge_list if edge_list[k]}

    return B

tie_decay_result = tie_decay_matrix(data, 1000)
# result = tie_decay_matrix(data, 3425100)
