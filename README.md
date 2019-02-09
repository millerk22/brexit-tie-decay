# brexit-tie-decay
MATH 276 Group looking into the Brexit data and applying tie decay framework to community detection.



## Data Format in .zip files

`(day)_(day)(month)_*.zip` is a zipped filed containing:
* `edge_dict.pkl` --  a Python `pickle` file that contains the dictionary
* `edge_list.csv` -- `csv` file that contains the interactions in chronological order 
* `nodes.txt` -- file that has mapping from node in the graph to the Twitter handle in the original dataset

## To open the Pickle file
Here's an example of how to get the dictionary from the pickle format (pretty straightforward...).
```python
  import pickle
  
  file_path = './edge_dict.pkl'
  edges_dict = pickle.load(open(file_path, "rb"))
  
```
