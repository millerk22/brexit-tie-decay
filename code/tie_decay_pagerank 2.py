import numpy as np
from scipy.sparse import csr_matrix
from tie_decay_edge_list import tie_decay_edge_list
import time
import os

# We have N twitter users and input an edge_list with only their interactions
# We also relabel these nodes from 0-(N-1)

def save_data(dirpath, prefix, data):
    np.save(os.path.join(dirpath, prefix+'_PR.npy'), data)

def tie_decay_PageRank(edge_dict, N, t_of_interest, alpha=10**-5, dt=86400,
                       s=0.85, maxerr=10**-4, save=True):
    """
    Computes the pagerank for each node at time t_of_interest
    Parameters:
    ----------
    (1) edge_dict: dictionary of tuples of time between pairs of nodes
    (2) N: number of nodes
    (3) t_of_interest: time of interest
    (4) alpha: decay coefficient
    (5) dt: timestep
    (6) s: probability of following an interaction
    (7) maxerr: max error of convergence
    ----------
    Note: Here the teleportation vector is assumed to be uniform.
    """
    t = 0
    B = None                   # Initialize tie-decay edgelist
    v = np.ones(N) / float(N)
    p = v.copy()               # Initialize PageRank vector
    PR = []                    # Initialize list of pagerank vector

    i = 0

    while t < t_of_interest:

        print ("now calculating pagerank at t={}".format(t+dt))

        # TODO:
        # Add a check here, if no interactions take place, the pagerank score
        # stays the same. No need to compute.

        start = time.time()

        # Update B(t)
        B = tie_decay_edge_list(edge_dict, t+dt, alpha=alpha,
                                edit_edgelist=True, prev=B, t_start=t)

        print ("Time taken to generate tie-decay-matrix: {}".format(time.time()-start))

        start = time.time()

        src_ind = [k[0] for k in B.keys()]
        dst_ind = [k[1] for k in B.keys()]
        weight = [v for v in B.values()]

        A = csr_matrix((weight, (src_ind, dst_ind)),shape=(N, N), dtype=np.float)
        # rsums = np.array(A.sum(1))[:,0]   # row sums
        rsums = A.sum(1).A1
        ri, ci = A.nonzero()     # the x, y coordinates of A's nonzero entries
        A.data /= rsums[ri]      # get the psuedo-inverse
        A = A.transpose()

        c = rsums.nonzero()             # bool array of sink states
        # v = np.ones(N) / float(N)     # teleportation

        print ("Time taken to compute A and c: {}".format(time.time()-start))

        start = time.time()
        # Compute pagerank r until we converge
        p0 = np.zeros(N)
        while np.sum(np.abs(p-p0)) > maxerr:
            p0 = p.copy()
            p = s * A.dot(p0) + (s * np.sum(p0[c]) + (1-s)) * v
            # Normalize after each iteration
            p = p/float(sum(p))

        print ("Time taken for PR to converge is {}".format(time.time()-start))

        t = t + dt
        i += 1
        PR.append(p)
        if save:
            save_data("YOUR_DIRECTORY","day"+str(i), p)

    # Return PageRank at each day
    return PR

if __name__ == "__main__":
    import pickle

    start = time.time()

    os.chdir("YOUR_DIRECTORY")
    with open('full_edge_dict.pkl','rb') as f:
        data = pickle.load(f)

    print ("Data loaded in {}".format(time.time()-start))

    start_time = time.time()
    PR = tie_decay_PageRank(data, 1322219, 86400*7*5)
    print("--- %s seconds ---" % (time.time() - start_time))

    import IPython
    IPython.embed()
