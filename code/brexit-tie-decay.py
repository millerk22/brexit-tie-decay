import os
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd

# os.chdir("/Users/qinyichen/Documents/brexit-tie-decay/Data/data_3_9jun")
# data = pd.read_csv("edge_list.csv")
os.chdir("/Users/qinyichen/Desktop")
data = pd.read_csv("tie_decay_test.csv")

source  = np.array(data["source_id"])
target  = np.array(data["target_id"])
N = max(max(source), max(target))      # N: number of elements

# G = nx.from_pandas_edgelist(data, "source_id", "target_id", "Weight")
# A = nx.adjacency_matrix(G)

# TODO: The following function assumes the timestamps in edge_list is a list
# of integers. Need to convert current timestamps  into minutes.

def tie_decay_matrix(edge_list, N, T, dt=1, alpha=0.1):

    """
    INPUT:
    (1) edge_list: an edge list that involves time, src, dst, type, weight
    (2) N: number of nodes
    (3) T: time of interest
    (4) dt: size of time step
    (5) alpha: decay coefficient

    OUTPUT:
    B(T): connection matrix at time T
    """

    t_start = edge_list["source_tweet_created_at"][0]
    # t_end = edge_list["source_tweet_created_at"][-1]

    # Initialize B at t = 0
    B = np.zeros((N, N))

    # Update B at different time
    for t in range(t_start, T, dt):
        # entries of B decay from t to t+dt
        B = B * np.exp(-alpha * dt)

        # Add the new interactions
        new_edges = edge_list[(edge_list["source_tweet_created_at"] >= t) &
                              (edge_list["source_tweet_created_at"] < t + dt)]

        for ind, row in new_edges.iterrows():
            B[row["source_id"]][row["target_id"]] += 1
            print (row["source_id"], row["target_id"])
            # TODO: Should we take weight into account?

    return B

import IPython
IPython.embed()
