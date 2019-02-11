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
    B_dict(T): dictionary for weight matrix
    """
    

    # Initialize B_dict dictionary for weight matrix
    B_dict = {}

    # Fill up B_dict according to edge_dict
    for key,value in edge_dict.items():
        #Initialize B_dict，in a structure of the dictionary of dictionaries to use the 
        #networkx function, nx.from_dict_of_dicts(B) later in main()
        #Format of B_dict is "dod = {0: {1: {'weight': 1}}} # single edge (0,1)"
        (i,j) = key #the pair of node
        
        B_dict[str(i)] = {}
        B_dict[str(i)][str(j)] = {}
        B_dict[str(i)][str(j)]["weight"] = 0
        
        #For each pair of nodes interacted,
        for k in range(len(value)-1):
            time_diff = T-value[k][0] #time difference between time of kth 
            #interaction and time of interest.
            
            #If time of interest is less than the current interaction time, 
            #go on to the next pair of node. Relying on the increasing 
            #order of time in j the time tuple here.
            if time_diff < 0:
                break
            
            decay = m.exp(-alpha*time_diff) #decay from current 
            #interaction to time of interest
            #if -alpha*time_diff != 0:
                #print(-alpha*time_diff)
            B_dict[str(i)][str(j)]["weight"] *= decay
            B_dict[str(i)][str(j)]["weight"] += value[k][1]*decay #value[k][1] is the 
            #weight of kthe interaction
            #if value[k][1]*decay !=0:
                #print(value[k][1]*decay)
    return B_dict


    
    
  
def main():
    #T input by hand, 1465369200 is the time of the end of the first week  
    B = tie_decay_matrix(edge_dict, 1465369200, 0.0000000001)
    
    #Turn dictionary B into a Graph
    G_B = nx.from_dict_of_dicts(B)
    #plotting(G_B)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
        
if __name__ == '__main__':
    main()

    
