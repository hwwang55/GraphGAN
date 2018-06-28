## GraphGAN

- This repository is the implementation of [GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611) ([arXiv](https://arxiv.org/abs/1711.08267)):
> GraphGAN: Graph Representation Learning With Generative Adversarial Nets  
Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Xing Xie, Minyi Guo  
32nd AAAI Conference on Artificial Intelligence, 2018


### Files in the folder
- data: training and test data
- pre_train: pre-trained node embeddings
  > Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
- results: evaluation results and the learned embeddings of the generator and the discriminator
- src: source codes


### Requirements
- tensorflow
- tqdm (for displaying the progress bar)
- pickle
- numpy
- sklearn


### Input
The input data should be an undirected graph in which node IDs start from *0* to *N-1* (*N* is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

##### txt file sample

```0	1```  
```3	2```  
```...```


### Basic usage
```mkdir cache```   
```cd src/GraphGAN```  
```python graph_gan.py```

