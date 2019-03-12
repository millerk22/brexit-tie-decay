# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:14:12 2019

@author: hexie
"""
import operator
import pickle
import numpy as np
from collections import Counter

def find_out_node_identity(myset,tags,values):
    output=list(myset.intersection(tags))
    print (len(output))
    count_R=0
    count_L=0
    try:
        for item in output:
            if values[item]=='0':
                count_L=count_L+1
            elif values[item]=='1':
                count_R=count_R+1
    except:
        ident=2
    
    if count_R>count_L:
        ident=1
    elif count_R<count_L:
        ident=0
    else:
        ident=2

    return ident



with open('hashtag.pkl', 'rb') as f:
    data = pickle.load(f)
## supursingly ? the top 10 nodes with highest pagerank have a lot of overlapping
node_id= open("nodes.txt","r")
d={}

for line in node_id:
    x=line.split(",")
    a=x[0]
    b=x[1]
    d[a]=b

value=data.values()
v=list(list(x) for x in value)
list_of_hashtags = [item for sublist in v for item in sublist]

count_of_hashtag=dict(Counter(list_of_hashtags))
sorted_c = sorted(count_of_hashtag.items(), key=operator.itemgetter(1))
meaningful=sorted_c[:200]

hashtag={}

for item in meaningful:
    print (item)
    answer = input("What is this node's type? (0-leave 1-stay 2-neutral 100-not found) ")
    type(answer)
    hashtag[item]=answer

with open('hastag_status.piclke','wb') as handle:
    pickle.dump(hashtag, handle,protocol=pickle.HIGHEST_PROTOCOL)

identity={}
for key in hashtag.keys():
    identity[key[0]]=hashtag[key]

with open('identity_of_hastag_top200.piclke','wb') as handle:
    pickle.dump(identity, handle,protocol=pickle.HIGHEST_PROTOCOL)
    
R_and_L = {key:val for key, val in identity.items() if val != '2'}
tags=set(R_and_L.keys())

with open('R&L.piclke','wb') as handle:
    pickle.dump(R_and_L, handle,protocol=pickle.HIGHEST_PROTOCOL)

node_status={}

for item in data.keys():
    vv=set(data[item])
    node_status[item]=find_out_node_identity(vv,tags,R_and_L)

non_neu={}
leave_nodes={}
remain_nodes={}
non_neutral_nodes={key:val for key, val in node_status.items() if val != 2}
leave_nodes={key:val for key, val in non_neutral_nodes.items() if val != 1}
remain_nodes={key:val for key, val in non_neutral_nodes.items() if val != 0}

with open('all_node_identity.piclke','wb') as handle:
    pickle.dump(node_status, handle,protocol=pickle.HIGHEST_PROTOCOL)


with open('leave_nodes.piclke','wb') as handle:
    pickle.dump(leave_nodes, handle,protocol=pickle.HIGHEST_PROTOCOL)


with open('remain_nodes.piclke','wb') as handle:
    pickle.dump(remain_nodes, handle,protocol=pickle.HIGHEST_PROTOCOL)
    

with open('nonneutral_nodes.piclke','wb') as handle:
    pickle.dump(non_neutral_nodes, handle,protocol=pickle.HIGHEST_PROTOCOL)
