"""
The class is used to evaluate the application of link prediction
"""

import numpy as np
from sklearn.metrics import accuracy_score
from src import utils


class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename  # each line: node_id1, node_id2
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)

    def eval_link_prediction(self):
        test_edges = utils.read_edges_from_file(self.test_filename)
        test_edges_neg = utils.read_edges_from_file(self.test_neg_filename)
        test_edges.extend(test_edges_neg)

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]]))
        test_label = np.array(score_res)
        median = np.median(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1
        accuracy = accuracy_score(true_label, test_label)

        return accuracy
