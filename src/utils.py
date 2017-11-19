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
        node_embed[int(emd[0]), :] = str_list_to_float(emd[1:])

    return node_embed

def generate_neg_links(embed_filename, train_filename, test_filename, ratio):
    """
    generate neg links for link prediction evaluation
    :param ratio: 0-1 float, control the ratio of choosing negative sample
    :return:
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


    # read the test dataset
    test_edges = read_edges_from_file(test_filename)
    # read the embeddings
    with open(embed_filename, mode="r") as f:
        lines = f.readlines()
        node_embeddings = [str_list_to_float(line.split()) for line in lines]
        n_node = len(node_embeddings)
        n_emd = len(node_embeddings[0]) - 1
    emd = np.random.rand(n_node, n_emd)
    for item in node_embeddings:  # generate a emd dict with the format {id:[emd]}
        emd[int(item[0])] = item[1:]

    # compute the inner product
    inner_prods = np.dot(emd, emd.transpose())
    # for each link in test dataset, find a neg sample
    neg_edges = []


    for i in range(len(test_edges)):
        r = np.random.rand()
        if r < ratio:
            edge = test_edges[i]
            start_node = edge[0]
            end_node = edge[1]
            neg_nodes = list(nodes.difference(set(linked_nodes[edge[0]])))
            neg_edge = [start_node, neg_nodes[np.argmin(inner_prods[start_node, neg_nodes])]]
            neg_edges.append(neg_edge)
        else:
            edge = test_edges[i]
            start_node = edge[0]
            end_node = edge[1]
            neg_nodes = list(nodes.difference(set(linked_nodes[edge[0]])))
            neg_node = np.random.choice(neg_nodes, size=1)[0]
            neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open("../data/link_prediction/CA-GrQc_neg.txt", "w+") as f:
        f.writelines(neg_edges_str)

def eval_link_prediction(embed_filename, train_filename, test_filename, test_neg_filename, log_dir, log_filename):
    """choose the topK after removing the positive training links
    Args:
        test_dataset: Nx3, array
    Returns:
        accuracy:
    """

    # read the embeddings
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    edges = train_edges + test_edges
    nodes = set()
    for edge in edges:
        nodes = nodes.union(set(edge))
    n_node = len(nodes)

    with open(embed_filename, mode="r") as f:
        lines = f.readlines()[1:]
        node_embeddings = [str_list_to_float(line.split()) for line in lines]
        n_emd = len(node_embeddings[0]) - 1
    emd = np.random.rand(n_node, n_emd)
    for item in node_embeddings:  # generate a emd dict with the format {id:[emd]}
        emd[int(item[0])] = item[1:]

    # construct the test dataset
    test_edges_pos = read_edges_from_file(test_filename)
    test_edges_neg = read_edges_from_file(test_neg_filename)
    test_edges_pos.extend(test_edges_neg)
    test_edges = test_edges_pos
    # may exists isolated point
    score_res = []
    for i in range(len(test_edges)):
        score_res.append(np.dot(emd[test_edges[i][0]], emd[test_edges[i][1]]))
    test_label = np.array(score_res)
    bar = np.median(test_label)  #
    ind_pos = test_label >= bar
    ind_neg = test_label < bar
    test_label[ind_pos] = 1
    test_label[ind_neg] = 0
    true_label = np.zeros(test_label.shape)
    true_label[0:len(true_label)//2] = 1

    precision = precision_score(true_label, test_label, average='micro')
    recall = recall_score(true_label, test_label, average='micro')
    f1 = f1_score(true_label, test_label, average='micro')
    accuracy = accuracy_score(true_label, test_label)
    # print("f1:", f1)
    # print("accuracy:", accuracy)
    # print(precision, recall, f1, accuracy)
    # gen_log = open(log_dir + log_filename, 'a+')
    # gen_log.write("accuracy" + '\t' + str(accuracy) + '\n')
    # gen_log.write("recall" + '\t' + str(recall) + '\n')
    # gen_log.write("precision" + '\t' + str(precision) + '\n')
    # gen_log.write("f1" + '\t' + str(f1) + '\n')
    # gen_log.flush()
    # gen_log.close()
    return accuracy, f1


if __name__ == "__main__":
    #generate_neg_links("../pre_train/GraphGAN/CA-GrQc_pair_gan_theory_pair_reward_2_back_0_57902_gen_120.txt", "../data/link_prediction/CA-GrQc_undirected_train.txt", "../data/link_prediction/CA-GrQc_test.txt", ratio=0.8)
    filenames = ["../pre_train/link_prediction/1.0/CA-GrQc_deepwalk_iters_9.emb", "../pre_train/link_prediction/1.0/CA-GrQc_LINE_iters_100.emb",
                 "../pre_train/link_prediction/1.0/CA-GrQc_node2vec_iters_1.emb", "../pre_train/link_prediction/1.0/CA-GrQc_struc2vec_iters_9.emb"]
    for filename in filenames:
        acc, f1 = eval_link_prediction(filename, "../data/link_prediction/CA-GrQc_undirected_train.txt",
                                       "../data/link_prediction/CA-GrQc_test.txt",
                                       "../data/link_prediction/CA-GrQc_neg.txt", "../log/", "CA-GrQc.txt")
        print(acc)
    #print(f1_res)
    # ids = [30681, 57565, 57902, 76134]
    # #ids = [57902]
    # for mode in ["gen", "dis"]:
    #     for id in ids:
    #         f1_res = []
    #         acc_res = []
    #         import tqdm
    #         filename = "../pre_train/link_prediction/1.0/CA-GrQc_deepwalk_iters_1.emb"
    #         acc, f1 = eval_link_prediction(filename, "../data/link_prediction/CA-GrQc_undirected_train.txt",
    #                                        "../data/link_prediction/CA-GrQc_test.txt",
    #                                        "../data/link_prediction/CA-GrQc_neg.txt", "../log/", "CA-GrQc.txt")
    #         f1_res.append(f1)
    #         acc_res.append(acc)
    #         for k in tqdm.tqdm(range(400)):
    #             i = 5*k
    #             #i = k
    #             filename = "../pre_train/GraphGAN/CA-GrQc_pair_gan_theory_pair_reward_2_back_0_" +  str(id) + "_"+ mode +"_" + str(i) + ".txt"
    #             acc, f1 = eval_link_prediction(filename, "../data/link_prediction/CA-GrQc_undirected_train.txt", "../data/link_prediction/CA-GrQc_test.txt", "../data/link_prediction/CA-GrQc_neg.txt", "../log/", "CA-GrQc.txt")
    #             f1_res.append(f1)
    #             acc_res.append(acc)
    #         #plt.plot([i for i in range(len(f1_res))], f1_res)
    #         #plt.show()
    #         f1_res_str = [str(x) + "\n" for x in f1_res]
    #         acc_res_str = [str(x) + "\n" for x in acc_res]
    #         f1_res_f = mode + "_5_f1_" + str(id) + ".txt"
    #         acc_res_f = mode + "_5_acc_" + str(id) + ".txt"
    #         with open(f1_res_f, "w+")  as f:
    #             f.writelines(f1_res_str)
    #         with open(acc_res_f, "w+")  as f:
    #             f.writelines(acc_res_str)





