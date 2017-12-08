import numpy as np
import copy
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def node_id_map(edges):
    """map the original node id to [0, node_num)

    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.union(set([node_set.index(edge[0])]))
        new_nodes = new_nodes.union(set([node_set.index(edge[1])]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes

def read_edges(train_filename, test_filename, mode=""):
    """read the data from the file

    Args:
        train_filename:
        test_filename:
    Returns:
        train_edges: list, whose element is a list like [node1, node2]
        test_edges: list, whose element is a list like [node1, node2]
        linked_nodes: dict, dict, <node_id, linked_nodes_id>, store the neighbor nodes of every node
    """

    linked_nodes = {}
    train_edges = read_edges_from_file(train_filename)
    if test_filename != "":
        test_edges = read_edges_from_file(test_filename)
    else:
        test_edges = []
    start_nodes = set()
    end_nodes = set()

    for edge in train_edges:
        start_nodes.add(edge[0])
        end_nodes.add(edge[1])
        if linked_nodes.get(edge[0]) is None:
            linked_nodes[edge[0]] = []
        if linked_nodes.get(edge[1]) is None:
            linked_nodes[edge[1]] = []
        linked_nodes[edge[0]].append(edge[1])
        linked_nodes[edge[1]].append(edge[0])

    for edge in test_edges:
        start_nodes.add(edge[0])
        end_nodes.add(edge[1])
        if linked_nodes.get(edge[0]) is None:
            linked_nodes[edge[0]] = []
        if linked_nodes.get(edge[1]) is None:
            linked_nodes[edge[1]] = []

    #  for recommendation, return user_num, item_num, linked_nodes
    #  for others, return node_num, linked_nodes
    if mode == "recommend":
        return len(start_nodes), len(end_nodes), linked_nodes
    else:
        return len(start_nodes.union(end_nodes)), linked_nodes

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]

    return edges

def get_max_degree(linked_nodes):
    """get the max degree of the network

    Args:
         linked_nodes: dict, <node_id, neighbor_nodes>
    Returns:
        None
    """

    max_degree = 0

    for key, val in linked_nodes.items():
        if len(val) > max_degree:
            max_degree = len(val)

    return max_degree

def read_emd(filename, n_node, n_embed):
    """use the pretrain node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = str_list_to_float(emd[1:])

    return node_embed

def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    linked_nodes = {}
    for edge in train_edges + test_edges:
        if linked_nodes.get(edge[0]) is None:
            linked_nodes[edge[0]] = []
        if linked_nodes.get(edge[1]) is None:
            linked_nodes[edge[1]] = []
        linked_nodes[edge[0]].append(edge[1])
        linked_nodes[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(linked_nodes))])

    # for each link in test dataset, find a neg sample
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        end_node = edge[1]
        neg_nodes = list(nodes.difference(set(linked_nodes[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)



