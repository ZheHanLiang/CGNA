# CNNA
## Introduction
This is the source code for *Unsupervised Social Networks Alignment via Cross Network Embedding*. 

## Execute
* *initial_data_processing.py* & *initial_data_processing_MC3.py*: Transforming the raw data into a suitable format for code to run.
* *main.py* : The main function to run the code.

### Example execute command
```
python initial_data_processing.py -n 5000 -g 5 -r 0.05 -d flickr
```
```
python initial_data_processing_MC3.py
```
```
python main.py -d flickr-flickr
```

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [networkx](http://networkx.github.io/)

## Data
### Raw datasets
We used four public datasets in the paper: 

* *[Last.fm](http://lfs.aminer.cn/lab-datasets/multi-sns/lastfm.tar.gz)*
* *[Flickr](http://lfs.aminer.cn/lab-datasets/multi-sns/livejournal.tar.gz)*
* *[MySpace](http://lfs.aminer.cn/lab-datasets/multi-sns/myspace.tar.gz)*

You can learn more about these three datasets from [here](https://www.aminer.cn/cosnet). 

* *[MC3](http://vacommunity.org/VAST+Challenge+2018+MC3)*

### Processed datasets
You can get the processed datasets in *-data/data_backup*.

If you want to use the back up data, please move the corresponding data to *-data/graph_edge*.

### User mapping
If you want to try the experiment across networks(*[Last.fm](http://lfs.aminer.cn/lab-datasets/multi-sns/lastfm.tar.gz)*, *[Flickr](http://lfs.aminer.cn/lab-datasets/multi-sns/livejournal.tar.gz)* and *[MySpace](http://lfs.aminer.cn/lab-datasets/multi-sns/myspace.tar.gz)*), we provide the processed real user mapping in *-data/graph_map*.
