"""
The class is used for evaluating the application of link prediction
"""

import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import utils


class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings[n_embed]
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_emd(embed_filename, n_node=n_node, n_embed=n_embed)

    def eval_link_prediction(self):
        """choose the topK after removing the positive training links
        Args:
            test_dataset:
        Returns:
            accuracy:
        """

        test_edges = utils.read_edges_from_file(self.test_filename)
        test_edges_neg = utils.read_edges_from_file(self.test_neg_filename)
        test_edges.extend(test_edges_neg)

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]]))
        test_label = np.array(score_res)
        bar = np.median(test_label)  #
        ind_pos = test_label >= bar
        ind_neg = test_label < bar
        test_label[ind_pos] = 1
        test_label[ind_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0:len(true_label) // 2] = 1
		
        accuracy = accuracy_score(true_label, test_label)

        return accuracy
