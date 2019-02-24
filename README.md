# brexit-tie-decay
MATH 276 Group looking into the Brexit data and applying tie decay framework to community detection.

* Please see: https://docs.google.com/presentation/d/1yo4kj23_5CfLAgU31bD9zBwSpgh2T9CkHLLb3ByNE-w/edit?usp=sharing for the Mini Presentation Google slides.
* Please see: https://www.overleaf.com/project/5c7232c116631219ceeaa676 for the Paper that is in progress.



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
