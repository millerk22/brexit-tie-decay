#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:11:23 2019

@author: xiaojing
"""

import os
import math as m
#import pandas as pd
import pickle
import time
import networkx as nx
start_time = time.time()
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from operator import itemgetter





os.chdir("/Users/xiaojing/Downloads/data_3_9jun-3")
with open('edge_dict.pkl', 'rb') as f:
    edge_dict = pickle.load(f)


def tie_decay_matrix(edge_dict, T, alpha):

    """ 
    INPUT:
    (1) edge_dict: dictionary of tuples of time between pairs of nodes
    (2) T: time of interest
    (3) alpha: decay coefficient
    OUTPUT:
    B_dict(T): list of tuples of weighted edges at time of interest
    """ 
    

    # Initialize B_dict list for weight matrix
    B_dict = []
    for key,value in edge_dict.items():
       
        (i,j) = key #the pair of node
        
        weight = 0
        for item in value:
            time_diff = T-item[0] 
            
            if time_diff < 0:
                break            
            decay = m.exp(-alpha*time_diff) 
            weight += item[1]*decay
        
            
        if weight != 0:   
            B_dict.append((i,j,weight))
              
    return B_dict

def plotting(G_B, nodelist): #code by qinyi
    print ("Now start plotting")
    edges,weights = zip(*nx.get_edge_attributes(G_B,'weight').items())
    color = ['r' if w > 10 ** -10 else 'b' for w in weights]
    width = [1 if w > 10 ** -10 else 0.1 for w in weights]
    pos = nx.spring_layout(G_B)

    plt.figure()
    nx.draw(G_B, pos, nodelist = nodelist,node_size=5, node_color='b', edgelist=edges, edge_color=color, width=width)
    plt.savefig("tie_decay_example.png")
    plt.close()
  
def main():
    #T input by hand, 1465369200 is the time of the end of the first week  
    T = [ 604800, 518400, 432000, 345600, 259200, 172800, 86400]
    B_list = [tie_decay_matrix(edge_dict, T[i], 0.000001) for i in range(6)]   
    B_sub = [sorted(B_list[i],key=itemgetter(2))[:3000] for i in range(6)] # a subgraph of most heavy weighted edges after each day
    
    G_B = nx.DiGraph() 
    G_B.add_weighted_edges_from(B_sub[1]) # subgraphs
   
    
    plotting(G_B,list(G_B))
    
    print("--- %s seconds ---" % (time.time() - start_time))
   
        
if __name__ == '__main__':
    main()



    
