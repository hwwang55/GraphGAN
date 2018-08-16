## GraphGAN

- This repository is the implementation of [GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611) ([arXiv](https://arxiv.org/abs/1711.08267)):
> GraphGAN: Graph Representation Learning With Generative Adversarial Nets  
Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Xing Xie, Minyi Guo  
32nd AAAI Conference on Artificial Intelligence, 2018

![](https://github.com/hwwang55/GraphGAN/blob/master/framework.jpg)

GraphGAN unifies two schools of graph representation learning methodologies: generative methods and discriminative methods, via adversarial training in a minimax game.
The generator is guided by the signals from the discriminator and improves its generating performance, while the discriminator is pushed by the generator to better distinguish ground truth from generated samples.
	


### Files in the folder
- `data/`: training and test data
- `pre_train/`: pre-trained node embeddings
  > Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
- `results/`: evaluation results and the learned embeddings of the generator and the discriminator
- `src/`: source codes


### Requirements
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.8.0
- tqdm == 4.23.4 (for displaying the progress bar)
- numpy == 1.14.3
- sklearn == 0.19.1


### Input format
The input data should be an undirected graph in which node IDs start from *0* to *N-1* (*N* is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

##### txt file sample

```0	1```  
```3	2```  
```...```


### Basic usage
```mkdir cache```   
```cd src/GraphGAN```  
```python graph_gan.py```

