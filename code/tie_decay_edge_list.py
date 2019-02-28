import numpy as np

def tie_decay_edge_list(edge_dict, T, alpha=10**-5, edit_edgelist=False,
                        prev=None, t_start=0):
    """
    INPUT:
    (1) edge_dict: dictionary of tuples of time between pairs of nodes
    (2) T: time of interest
    (3) alpha: decay coefficient
    (4) t_start: start of time (assume edge_dict start with that)
    (5) prev: previous B

    OUTPUT:
    B_dict: dictionary for edge with weight
    """

    # Initialize B_dict
    B_dict = {}

    if prev:
        for k, v in prev.items():
            B_dict[k] = np.exp(-alpha * (T-t_start)) * v

    if edit_edgelist == False:
        # Fill up B_dict according to edge_dict
        for k, v in edge_dict.items():
            # If our time of interest is before the first event
            if T < v[0][0]:
                continue

            # For each interaction, calculate the connection contribution
            for i in range(len(v)):
                # diff btw time of ith interaction and time of interest
                time_diff = T - v[i][0]

                # If time of interest is less than the current interaction time,
                # no need to check the remaining interactions
                if time_diff < 0:
                    break

                decay = np.exp(-alpha * time_diff) # decay factor
                if i == 0:
                    B_dict[k] = v[i][1] * decay
                else:
                    B_dict[k] += v[i][1] * decay
    else:
        # Fill up B_dict according to edge_dict
        for k in edge_dict.keys():
            # If our time of interest is before the first event
            if T < edge_dict[k][0][0]:
                continue

            # For each interaction, calculate the connection contribution
            while len(edge_dict[k]) > 0 and T >= edge_dict[k][0][0]:
                decay = np.exp(-alpha * (T-edge_dict[k][0][0]))
                try:
                    B_dict[k] += edge_dict[k][0][1] * decay
                except KeyError:
                    B_dict[k] = edge_dict[k][0][1] * decay
                # pop the edge out of the edge_dict
                edge_dict[k].pop(0)

        # Clean up the edge_dict if all edges has been considered
        for k in list(edge_dict):
            if len(edge_dict[k]) == 0:
                del edge_dict[k]

    return B_dict

if __name__ == "__main__":
    # import os
    import time
    import pickle

    # os.chdir("/Users/qinyichen/Documents/brexit-tie-decay/Data/data_3_9jun")
    with open('edge_dict.pkl','rb') as f:
        data = pickle.load(f)

    A1 = tie_decay_edge_list(data, 200000)
    A2 = tie_decay_edge_list(data, 400000)

    start_time = time.time()
    B1 = tie_decay_edge_list(data, 200000, edit_edgelist=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    B2 = tie_decay_edge_list(data, 400000, edit_edgelist=True, prev=B1, t_start=200000)
    print("--- %s seconds ---" % (time.time() - start_time))

    import IPython
    IPython.embed()
