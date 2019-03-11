edge_lists.zip -- contains edge lists in csv files that have columns 
		source_tweet_created_at, source, target, weight, Red

full_data.zip -- full_edge_dict.pkl = file which is a pickled python dictionary containing all of the edge interactions over the whole time period that we have data for. (~215 MB in pickled format).
		nodes.txt = node id to Twitter name conversions

data_3_9jun_dict.zip -- OLD, OUTDATED zipped file containing smaller scale data, just in terms of the isolated week of Jun 3 - Jun 9, 2016


hashtag.pkl -- python pickle file that contains python dictionary of (key, value) pairs as (source, {source_hashtags}). 'source' is the Twiter handle and the value is a set of the unique hashtags that the person has used throughout the timeline of our dataset. 

w[n].csv -- Nodes of week n, followed by the subcommunities find by the mapequation, with remain neutral or leave assigned to the subcommunities.
