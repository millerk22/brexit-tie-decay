# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:45:16 2019

@author: hexie
"""
import numpy as np
import operator
import pickle
import matplotlib.pyplot as plt
 
# Data to plot


node_id= open("nodes.txt","r")
idmap={}

for line in node_id:
    x=line.split(",")
    a=x[0]
    b=x[1]
    b=str(b).strip()
    idmap[a]=b

with open('week4.pickle', 'rb') as f:
    data = pickle.load(f)
    
with open('week4_degree.pickle','rb') as f:
    degree=pickle.load(f)

with open('all_node_identity.piclke','rb') as f:
    node_value=pickle.load(f)

num_of_com=max(data.values())

communities={}

count_N=0
count_R=0
count_L=0

for comm in range(num_of_com):
    mynodes={key:val for key, val in data.items() if val == comm}
    mynodes=list(mynodes.keys())
    mydegree=list([degree[x] for x in mynodes])
    mydict=dict(zip(mynodes,mydegree))
    sorted_c = sorted(mydict.items(), key=operator.itemgetter(1),reverse=True)
    try: 
        print (idmap[str(sorted_c[0][0])])
        ident=node_value[idmap[str(sorted_c[0][0])]]
        print (ident)
    except:
        # if a node dont have hashtag- it is a neutral node
        ident=2
#    count_R=0
#    count_L=0
#    
#    try:
#        for item in sorted_c[:10]:
#            if node_value[item]==0:
#                count_L=count_L+1
#            elif node_value[item]==1:
#                count_R=count_R+1
#    except:
#        ident=2
#    
#    if count_R>count_L:
#        ident=1
#    elif count_R<count_L:
#        ident=0
#    else:
#        ident=2        
    communities[comm]=ident
    if ident==2 :
        count_N=count_N+len(mynodes)
    elif ident==1 :
        count_R=count_R+len(mynodes)
    elif ident==0 :
        count_L=count_L+len(mynodes)
        
labels = 'Neutral', 'Leave', 'Remain'
sizes = [count_N, count_L, count_R]
colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0)  # explode 1st slice
 
plt.clf()
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
plt.savefig('week4_pie.png')
    