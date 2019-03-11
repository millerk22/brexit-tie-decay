import collections
import sys
import pickle
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# where the code tie_decay_edge_list.py is locatex
os.chdir("/Users/Kevin/Desktop/UCLA/2019/276/Project/brexit-tie-decay/code")
import tie_decay_edge_list as tdel

def get_hashtag_dict(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d

def check_node_time(s, td_time):
    print('Checking if node %d is in the tie decay graph at this time %f' % (s, td_time))
    with open('/Users/Kevin/Desktop/UCLA/2019/276/Project/full_data_smaller/full_edge_dict.pkl', 'rb') as f:
        d = pickle.load(f)

    # calculate the tie decay network at edge dictionary
    print('Calculating tie decay matrix')
    B = tdel.tie_decay_edge_list(d, td_time, edit_edgelist=True)
    edges = list(B.keys())
    vals = list(B.values())
    keys1, keys2 = zip(*edges)
    keys1 = list(keys1)
    try:
        idx = keys1.index(s)
        return 1
    except:
        print('node %d is not in the graph at this time' % s)
        return 0

def check_node_names(n):
    df = pd.read_csv('/Users/Kevin/Desktop/UCLA/2019/276/Project/full_data_smaller/nodes.txt')
    vals = df.values
    nodes, names = list(vals[:,0]), list(vals[:,1])
    return [names[nodes.index(i)] for i in n]



'''
Original code by @dgleich, obtained from github.

Changes done by Kevin Miller 3/3/2019

'''
# parameters for actions
community = '0'  # '0' = Leave, '1' = Remain, '2' = Neutral
week = 3
plot = False
ipy = False
verbose = False
diff_calc= False


# community name list for plotting
comm = ['Leave', 'Remain', 'Neutral']

# constants for later use
sec_per_day = 60*60*24
sat_second = sec_per_day*7
tot_seconds = sat_second*5

# what time you'd like to calculate the tie decay network at
td_time = week*sat_second
print('Calculating PPR for %s of Saturday %d\n\n' % (comm[int(community)], week))

fname = './G_%d.pkl' % td_time
if os.path.isfile(fname): # if have already translated this graph to node edge list then load it
    print('Found the file')
    with open(fname, 'rb') as f:
        Gvol, G, B = pickle.load(f)

else:
    print('Graph for time %d not present, so calculating' % td_time)
    tic = time.clock()
    # load the edge dictionary, for me is in full_data_smaller directory
    with open('/Users/Kevin/Desktop/UCLA/2019/276/Project/full_data_smaller/full_edge_dict.pkl', 'rb') as f:
        d = pickle.load(f)
    t = time.clock() - tic
    print('%f seconds to load edge_dict' % t)

    # calculate the tie decay network at edge dictionary
    B = tdel.tie_decay_edge_list(d, td_time, edit_edgelist=True)
    edges = list(B.keys())
    vals = list(B.values())
    keys1, keys2 = zip(*edges)
    keys1, keys2 = np.array(keys1), np.array(keys2)
    X = np.vstack((keys2, np.array(vals)))
    G = {}
    k1 = np.unique(keys1)
    tic = time.clock()
    for u in k1:
        u_out = X[:, keys1 == u]
        G[u] = (u_out[0, :], u_out[1, :])

    t = time.clock() - tic
    print('%f seconds to compute new network dictionary...' % t)
    Gvol = sum(vals)

    with open(fname, 'wb') as f:
        pickle.dump((Gvol, G, B), f)


##### Need to get the seed nodes for the specific communities

s_d = np.load('/Users/Kevin/Desktop/UCLA/2019/276/Project/brexit-tie-decay/Data/seed.npy').item()


seed = [k for k,v in s_d.items() if v == community]

for s in seed:
    try:
        nbrs, weights = G[s]
    except:
        print('key %d is not in the graph... must happen at future time. Removing from seed list' % s)
        seed.pop(seed.index(s))


nodes = pd.read_csv('/Users/Kevin/Desktop/UCLA/2019/276/Project/full_data_smaller/nodes.txt')
nodeid2name = dict(zip(nodes['node_id'], nodes['name']))
#names = [nodeid2name[n] for n in seed]
top_nodes, com = zip(*s_d.items())
top_nodes = np.array(list(top_nodes))
com = np.array(list(com))
print(top_nodes)
print(com)
s_d_names = [nodeid2name[n] for n in top_nodes]
labeled_nodes = {}
for c in ['0','1','2']:
    print('Community %s:' % comm[int(c)])
    c_nodes = top_nodes[np.where(com == c)[0]]
    labeled_nodes[c] = c_nodes
    print(str([nodeid2name[n] for n in c_nodes]))
print('\n')
#print('Here are the names in the seed set: Saturday %d, community %s' %(week, comm[int(community)]))






# G is graph as dictionary-of-(list, list)
alpha=0.99
tol=0.01

BESTSETS = {}
max_iter = 5
old_seed = [0]
it = 0
while it < max_iter and sorted(old_seed) != sorted(seed):
    old_seed = seed
    # PPR calculation for the given seed set
    tic = time.clock()
    x = {} # Store x, r as dictionaries
    r = {} # initialize residual
    Q = collections.deque() # initialize queue
    for s in seed:
      r[s] = 1.0/len(seed)
      Q.append(s)
    while len(Q) > 0:
      v = Q.popleft() # v has r[v] > tol*deg(v)
      if v not in x: x[v] = 0.
      x[v] += (1-alpha)*r[v]
      nbrs, weights = G[v]
      v_s = sum(weights)
      mass = alpha*r[v]/(2*v_s)  # the same for each nbr of v
      for i in range(len(nbrs)): # for neighbors of u
        u = int(nbrs[i])
        #print('%d, %d: %f' % (v, ))
        assert u is not v, "contact dgleich@purdue.edu for self-links"
        if u not in r: r[u] = 0.
        try:
            if r[u] < sum(G[u][1])*tol and \
               r[u] + weights[i]*mass >= sum(G[u][1])*tol:
               Q.append(u) # add u to queue if large
            r[u] = r[u] + weights[i]*mass
        except:
            continue
      r[v] = mass*v_s
      if r[v] >= v_s*tol: Q.append(v)
    if verbose:
        print( str(x))


    t = time.clock() - tic
    print('%f seconds to do PPR' % t)




    check = True
    #############################
    ## Calculate the best cluster, based on conductance. Need to do in terms of B
    # Find cluster, first normalize by degree
    for v in x: x[v] = x[v]/sum(G[v][1])
    # now sort x's keys by value, decreasing
    sv = sorted(x.items(), key=lambda x: x[1], reverse=True)
    S = set()
    volS = 0.
    cutS = 0.
    bestcond = 1.
    bestset = sv[0]
    conductances = []
    print('Gvol = %f' % Gvol)
    for p in sv:
      s = p[0] # get the vertex

      #volS += sum(G[s][1]) # add degree to volume
      volS += sum(G[s][1]) # degree will be out_deg + in_deg. first add out_deg
      for i in range(len(G[s][0])): # go through neighbors of s
        n = G[s][0][i]
        # Get both directed edges from B
        w = G[s][1][i]  # out weight
        try:            # add edge weight from n to s, if exists
            w_in = B[(n,s)]
            w += w_in
            volS += w_in  # add this edge's contribution to in_deg, added to volS
        except:
            #print('edge (%d,%d) not in B' % (n,s))
            continue

        if n in S:    # if neighbor n is already in set S, remove the total edge weight w from the cut
          cutS -= w
          #if check: print('sub %d volS, cutS = %f, %f' % (G[s][0][i], volS, cutS))
        else:   # if node NOT in set S, add the total edge weight w from the cut
          cutS += w
          #if check: print('add %d volS, cutS = %f, %f' % (G[s][0][i], volS, cutS))
      #if verbose: print( "v: %4i  cut: %4f  vol: %4f"%(s, cutS,volS) )
      #print( "v: %4i  cut: %4f  vol: %4f"%(s, cutS,volS) )
      S.add(s)
      conductance = cutS/min(volS,Gvol-volS)
      conductances.append(conductance)
      #print(conductance)
      if conductance <= bestcond and conductance != 0:
        bestcond = conductance
        bestset = set(S) # make a copy
        #print(bestset)
    print( "Best set conductance: %f"%(bestcond) )
    print( "  bestset, iter %d = %s" % (it, str(bestset)))

    print("\nPrevious seed set: %s", str(seed))
    #seed.extend(list(bestset))
    seed = np.setdiff1d(np.union1d(seed, list(bestset)), labeled_nodes['2'])
    print("New seed set: %s", str(seed))
    print('\n')

    it += 1


print('Now lets do a visual check of the Twitter handles of our final community')
#print('loading names/nodes dict')
#nodes = pd.read_csv('/Users/Kevin/Desktop/UCLA/2019/276/Project/full_data_smaller/nodes.txt')
#nodeid2name = dict(zip(nodes['node_id'], nodes['name']))
names = [nodeid2name[n] for n in seed]
print('Here are the names in the community: Saturday %d, community %s' %(week, comm[int(community)]))
print(str(names))




hash_dict = get_hashtag_dict('/Users/Kevin/Desktop/UCLA/2019/276/Project/hashtag.pkl')
import IPython
IPython.embed()





if diff_calc:
    print('\n\nDoing a different calculation of best set \n')
    C = list(zip(conductances[:-2], conductances[1:-1], conductances[2:]))
    l_mzrs = []
    for i in range(len(C)):
        prev, curr, next = C[i]
        if curr <= prev and curr <= next:
            l_mzrs.append(i+1) # gives the index in the original conductances list
    print(conductances)
    print(l_mzrs)
    print()


if verbose:
    nodes, pr = zip(*sv)
    nodes = list(nodes)
    node_names = check_node_names(nodes)
    print(str([len(G[u][0]) for u in nodes]))
    print('%s : %s' %(str(seed), str([len(G[u][0]) for u in seed])))
    out_degs = [sum(G[u][1]) for u in nodes]

    '''
    plt.plot(range(len(nodes)), pr, linewidth=1.5)
    plt.xticks(range(len(nodes)), node_names, rotation= 45)
    plt.title('PPR for %s' % comm[int(community)])
    plt.show()

    plt.plot(range(len(nodes)), pr, linewidth=1.5)
    plt.xticks(range(len(nodes)), nodes, rotation= 45)
    plt.title('PPR for %s' % comm[int(community)])
    plt.text(2, 0.7*max(pr), str(bestset))
    plt.show()
    '''

    plt.plot(range(len(conductances)), conductances)
    plt.scatter(l_mzrs, [conductances[i] for i in l_mzrs], c='r',marker='*', label='local mins')
    plt.title('Conductance plot')
    plt.legend(loc='best')
    plt.xlabel('# of nodes')
    plt.ylabel('conductance')
    plt.show()




if ipy:
    import IPython
    IPython.embed()


if plot:

    import networkx as nx
    import matplotlib.pyplot as plt



    G1 = {}
    nc = {}
    for k in seed:
        G1[k] = {}
        nbrs, w = G[k]
        for i in range(len(nbrs)):
            G1[k][nbrs[i]] = {'weight' : w[i]}
    #nc = ['b' if i+1 in bestset else 'r' for i in range(len(x.keys()))]
    #nx.draw_kamada_kawai(nx.to_networkx_graph(G1), with_labels=True, node_color=nc)
    graph = nx.to_networkx_graph(G1)
    nodes = list(graph.nodes())
    nc = ['r']*len(nodes)
    for i in range(len(nodes)):
        if nodes[i] in seed:
            nc[i] = 'g'
        if nodes[i] in bestset:
            nc[i] = 'b'
    nx.draw_kamada_kawai(graph, with_labels=False, node_color= nc)
    plt.show()
