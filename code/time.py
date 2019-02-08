import os
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd

# os.chdir("/Users/qinyichen/Documents/brexit-tie-decay/Data/data_3_9jun")
# data = pd.read_csv("edge_list.csv")
os.chdir("data_3_9jun\data_3_9jun")
data = pd.read_csv("edge_list.csv")
#
#time_data  = data["source_tweet_created_at"].to_string()
time_data=data['source_tweet_created_at'].astype(str).values.tolist()

import time
import datetime
#format = "%d/%m/%Y  %H:%M"
format="%Y-%m-%d %H:%M:%S"

output=np.zeros(data.shape[0])

for i in range(data.shape[0]):
    s=str(time_data[i])
    minutes = time.mktime(datetime.datetime.strptime(s, format).timetuple())
    output[i]=minutes
    
output=pd.DataFrame(output, columns=['int_time_in_second'])
output_time=pd.concat([output,data],axis=1)
output_time.to_csv("time_in_sec.csv")