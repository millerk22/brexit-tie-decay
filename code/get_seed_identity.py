# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:38:57 2019

@author: hexie
"""
#import pickle
import numpy as np
path="TopUsers/"

def get_top_X_Users(data_root, X=10, A=0, B=35):
    # get_top_X_Users return a np array with the union of ids of X top users
    # from specified day A to specified day B.
    # if not otherwise specified it only take the first 10 users
    old=[]
    for i in range (B-A):
        data= np.load(data_root+"day{}_users.npy".format(A+i+1)).item()
        data={key: data[key] for key in sorted(data, key=data.get, reverse=True)[:X]}
        for key in data:
            old.append(key)
    return old

K=list(set(get_top_X_Users(path,3)))

#with open('full_edge_dict.pkl', 'rb') as f:
#    data = pickle.load(f)
## supursingly ? the top 10 nodes with highest pagerank have a lot of overlapping
node_id= open("nodes.txt","r")
d={}

for line in node_id:
    x=line.split(",")
    a=x[0]
    b=x[1]
    d[a]=b

seed={}

#for item in K:
#    print (d[str(item)])
#    answer = input("What is this node's type? (0-leave 1-stay 2-neutral) ")
#    type(answer)
#    seed[item]=answer