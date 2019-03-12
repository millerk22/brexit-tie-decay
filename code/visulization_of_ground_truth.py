# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:47:09 2019

@author: hexie
"""
import numpy as np
import operator
import pickle
import matplotlib.pyplot as plt

def draw_pie(count_N,count_L,count_R,name):
    labels = 'Neutral', 'Leave', 'Remain'
    sizes = [count_N, count_L, count_R]
    colors = ['red', 'lightcoral', 'lightskyblue']
    explode = (0, 0, 0)  # explode 1st slice
     
    plt.clf()
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
     
    plt.axis('equal')
    plt.savefig('week'+str(name)+'ground_truth_pie.png')
    
def return_input(mdict, node_value,idmap):
    keys=list(mdict.keys())
    values=[]
    for x in keys:
        try:
            values.append(node_value[idmap[str(x)]])
        except:
            values.append(2)
    mydict=dict(zip(keys,values))
    n={key:val for key, val in mydict.items() if val == 2}
    l={key:val for key, val in mydict.items() if val == 0}
    r={key:val for key, val in mydict.items() if val == 1}
    
    return len(list(n.keys())),len(list(l.keys())),len(list(r.keys()))

node_id= open("nodes.txt","r")
idmap={}

for line in node_id:
    x=line.split(",")
    a=x[0]
    b=x[1]
    b=str(b).strip()
    idmap[a]=b

with open('all_node_identity.piclke','rb') as f:
    node_value=pickle.load(f)
    
with open('week1_degree.pickle','rb') as f:
    week1=pickle.load(f)

with open('week2_degree.pickle','rb') as f:
    week2=pickle.load(f)
    
with open('week3_degree.pickle','rb') as f:
    week3=pickle.load(f)
    
with open('week4_degree.pickle','rb') as f:
    week4=pickle.load(f)
    
with open('week5_degree.pickle','rb') as f:
    week5=pickle.load(f)
    
n,l,r= return_input(week1,node_value,idmap)
draw_pie(n,l,r,1)
n,l,r= return_input(week2,node_value,idmap)
draw_pie(n,l,r,2)
n,l,r= return_input(week3,node_value,idmap)
draw_pie(n,l,r,3)
n,l,r= return_input(week4,node_value,idmap)
draw_pie(n,l,r,4)
n,l,r= return_input(week5,node_value,idmap)
draw_pie(n,l,r,5)
