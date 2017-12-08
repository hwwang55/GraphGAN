import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import generator
import discriminator
import utils
import collections
import tqdm
import copy
import numpy as np
import tensorflow as tf
import time
import os
import multiprocessing
import config
import evaluation.eval_link_prediction as elp


class GraphGan(object):
    def __init__(self):
        """initialize the parameters, prepare the data and build the network"""

        self.n_node, self.linked_nodes = utils.read_edges(config.train_filename, config.test_filename)
        self.root_nodes = [i for i in range(self.n_node)]
        self.discriminator = None
        self.generator = None
        assert self.n_node == config.n_node
        print("start reading initial embeddings")
        # read the initial embeddings
        self.node_embed_init_d = utils.read_emd(filename=config.pretrain_emd_filename_d, n_node=config.n_node, n_embed=config.n_embed)
        self.node_embed_init_g = utils.read_emd(filename=config.pretrain_emd_filename_g, n_node=config.n_node, n_embed=config.n_embed)
        print("finish reading initial embeddings")
        # use the BFS to construct the trees
        print("Constructing Trees")
        if config.app == "recommendation":
            self.mul_construct_trees_for_recommend(self.user_nodes)
        else:  # classification
            self.mul_construct_trees(self.root_nodes)
        config.max_degree = utils.get_max_degree(self.linked_nodes)
        self.build_gan()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)



    def sample_for_gan(self, root, tree, sample_num, all_score, sample_for_dis):
        """ sample the nodes from the tree

        Args:
            root: int, root, the query
            tree: dict, tree information
            sample_num: the number of the desired sampling nodes
            all_score: pre-computed score matrix, speed the sample process
            sample_for_dis: bool, indicates it is sampling for generator or discriminator
        Returns:
            sample: list, include the index of the sampling nodes
        """

        assert sample_for_dis in [False, True]

        sample = []
        self.trace = []
        n = 0

        while len(sample) < sample_num:
            node_select = root
            node_father = -1
            self.trace.append([])
            flag = 1 #
            self.trace[n].append(node_select)
            while True:
                if flag == 1:
                    node_neighbor = tree[node_select][1:]
                else:
                    node_neighbor = tree[node_select]

                flag = 0
                if node_neighbor == []:  # the tree only has the root
                    return sample
                if  sample_for_dis == True:  # only sample the negative examples for discriminator, thus should exclude the root node tobe sampled
                    if node_neighbor == [root]:
                        return sample
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                if len(node_neighbor) == 0:
                    return sample
                prob = all_score[node_select, node_neighbor]
                prob = self.softmax(prob)
                if np.sum(prob) - 1 < 0.001:
                    pass
                else:
                    print(prob)
                node_check = np.random.choice(node_neighbor, size=1, p=prob)[0]
                self.trace[n].append(node_check)
                if node_check == node_father:
                    sample.append(node_select)
                    break
                node_father = node_select
                node_select = node_check
            n = n + 1
        return sample
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # for numberation stablity
        return e_x / e_x.sum()
        
    def mul_construct_trees(self, nodes):
        """use the multiprocessing to speed the process of constructing trees

        Args:
            nodes: list, the root of the trees
        """

        if config.use_mul:
            t1 = time.time()
            cores = multiprocessing.cpu_count() // 2
            pool = multiprocessing.Pool(cores)
            new_nodes = []
            node_per_core = self.n_node // cores
            for i in range(cores):
                if i != cores - 1:
                    new_nodes.append(nodes[i*node_per_core:(i+1)*node_per_core])
                else:
                    new_nodes.append(nodes[i*node_per_core:])

            self.trees = {}
            trees_result = pool.map(self.construct_tree, new_nodes)
            for tree in trees_result:
                self.trees.update(tree)
            t2 = time.time()
            print(t2-t1)
        else:
            self.trees = self.construct_tree(nodes)
        # serialized the trees to the disk
        print("Dump the trees to the disk")

    def construct_tree(self, nodes):
        """use the BFS algorithm to construct the trees

        Works OK.
        test case: [[0,1],[0,2],[1,3],[1,4],[2,4],[3,5]]
        "Node": [father, children], if node is the root, then the father is itself.
        Args:
            nodes:
        Returns:
            trees: dict, <key, value>:<node_id, {dict(store the neighbor nodes)}>
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            tmp = copy.copy(self.linked_nodes[root])
            trees[root][root] = [root] + tmp
            if len(tmp) == 0:  # isolated user
                continue
            queue = collections.deque(tmp)  # the nodes in this queue all are items
            for x in tmp:
                trees[root][x] = [root]
            used_nodes = set(tmp)
            used_nodes.add(root)

            while len(queue) > 0:
                cur_node = queue.pop()
                used_nodes.add(cur_node)
                for sub_node in self.linked_nodes[cur_node]:
                    if sub_node not in used_nodes:
                        queue.appendleft(sub_node)
                        used_nodes.add(sub_node)
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
        return trees

    def build_discriminator(self):
        """initialize the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def build_generator(self):
        """initialize the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def build_gan(self):
        """build the gan network"""
        self.build_generator()
        self.build_discriminator()
        # to save the model information every fixed number training epochs
        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        # add the saver
        self.saver = tf.train.Saver()

    def generate_for_d(self):
        """Generate the pos and neg samples for the Discriminator, and record them in the txt file"""

        self.samples_rel = []
        self.samples_q = []
        self.samples_label = []
        all_score = self.sess.run(self.generator.all_score)
        for u in self.root_nodes:
            if np.random.rand() < config.update_ratio:  #
                pos = self.linked_nodes[u]  # pos samples
                if len(pos) < 1:
                    continue
                self.samples_rel.extend(pos)
                self.samples_label.extend(len(pos) * [1])
                self.samples_q.extend(len(pos) * [u])
                neg = self.sample_for_gan(u, self.trees[u], len(pos), all_score, sample_for_dis=True)
                if len(neg) < len(pos):
                    continue
                self.samples_rel.extend(neg)
                self.samples_label.extend(len(neg)*[0])
                self.samples_q.extend(len(pos) * [u])

    def get_batch_data(self, index, size):
        """ take out sample of size from the samples
        Args:
            index: the start index
            size: the number of the batch, may not equal to batch size
        """

        q_node = self.samples_q[index:index+size]
        rel_node = self.samples_rel[index:index+size]
        label = self.samples_label[index:index+size]

        return q_node, rel_node, label

    def train_gan(self):
        """train the whole graph gan network"""

        #  restore the model
        #  check if there exists checkpoint, if true, load it
        ckpt = tf.train.get_checkpoint_state(config.model_log)
        if ckpt and ckpt.model_checkpoint_path and config.load_model:
            print("Load the checkpoint: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Evaluation")
        self.write_emb_to_txt()
        self.eval_test()  # evaluation
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            #  save the model
            if epoch % config.save_steps == 0 and epoch > 0:
                self.saver.save(self.sess, config.model_log + "model.ckpt")


            for d_epoch in tqdm.tqdm(range(config.max_epochs_dis)):
                if d_epoch % config.gen_for_d_iters == 0:  # every gen_for_d_iters round, we generate new data
                    self.generate_for_d()
                train_size = len(self.samples_q)
                #  traverse the whole training dataset sequentially, train the discriminator
                index = 0
                while True:
                    if index > train_size:
                        break
                    if index + config.batch_size_dis <= train_size + 1:
                        input_q_node, input_rel_node, input_label = self.get_batch_data(index, config.batch_size_dis)
                    else:
                        input_q_node, input_rel_node, input_label = self.get_batch_data(index, train_size - index)
                    index += config.batch_size_dis
                    _, loss = self.sess.run([self.discriminator.d_updates, self.discriminator.pre_loss], {self.discriminator.q_node: np.array(input_q_node),\
                                      self.discriminator.rel_node: np.array(input_rel_node), self.discriminator.label: np.array(input_label)})

            for g_epoch in tqdm.tqdm(range(config.max_epochs_gen)):
                cnt = 0
                root_nodes = []  # just for record how many trees that have been traversed
                rel_nodes = []  # the sample nodes of the root node
                root_nodes_gen = []  # root node feeds into the network, same length as rel_node
                trace = []  # the trace when sampling the nodes, from the root to leaf  bach to leaf's father. e.g.: 0 - 1 - 2 -1
                all_score = self.sess.run(self.generator.all_score)  # compute the score for computing the probability when sampling nodes
                for root_node in tqdm.tqdm(self.root_nodes, mininterval=3):  # random update trees
                    if np.random.rand() < config.update_ratio:
                        # sample the nodes according to our method.
                        # feed the reward from the discriminator and the sampled nodes to the generator.
                        if cnt % config.gen_update_iter == 0 and cnt > 0:
                            # generate update pairs along the path, [q_node, rel_node]
                            pairs = list(map(self.generate_window_pairs, trace))  # [[], []] each list contains the pairs along the same path
                            q_node_gen = []
                            rel_node_gen = []
                            for ii in range(len(pairs)):
                                path_pairs = pairs[ii]
                                for pair in path_pairs:
                                    q_node_gen.append(pair[0])
                                    rel_node_gen.append(pair[1])
                            reward_gen, node_embed = self.sess.run([self.discriminator.score, self.discriminator.node_embed],
                                                       {self.discriminator.q_node: np.array(q_node_gen),
                                                        self.discriminator.rel_node: np.array(rel_node_gen)})

                            feed_dict = {self.generator.q_node: np.array(q_node_gen), self.generator.rel_node: np.array(rel_node_gen),
                                         self.generator.reward: reward_gen}
                            _, loss, prob = self.sess.run([self.generator.gan_updates, self.generator.gan_loss, self.generator.i_prob],
                                                          feed_dict=feed_dict)
                            all_score = self.sess.run(self.generator.all_score)
                            root_nodes = []
                            rel_nodes = []
                            root_nodes_gen = []
                            trace = []
                            cnt = 0
                        sample = self.sample_for_gan(root_node, self.trees[root_node], config.n_sample_gen, all_score, sample_for_dis=False)
                        if len(sample) < config.n_sample_gen:
                            cnt = len(root_nodes)
                            continue
                        root_nodes.append(root_node)
                        root_nodes_gen.extend(len(sample)*[root_node])
                        rel_nodes.extend(sample)
                        trace.extend(self.trace)
                        cnt = cnt + 1

            print("Evaluation")
            self.write_emb_to_txt()
            self.eval_test()  # evaluation

    def generate_window_pairs(self, sample_path):
        """
        given a sample path list from root to a sampled node, generate all the pairs corresponding to the windows size
        e.g.: [1, 0, 2, 4, 2], window_size = 2 -> [1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]
        :param sample_path:
        :return:
        """

        sample_path = sample_path[:-1]
        pairs = []

        for i in range(len(sample_path)):
            center_node = sample_path[i]
            for j in range(max(i-config.window_size, 0), min(i+config.window_size+1, len(sample_path))):
                if i == j:
                    continue
                node = sample_path[j]
                pairs.append([center_node, node])

        return pairs

    def padding_neighbor(self, neighbor):
        return neighbor + (config.max_degree - len(neighbor)) * [0]

    def save_emb(self, node_embed, filename):
        np.savetxt(filename, node_embed, fmt="%10.5f", delimiter='\t')

    def write_emb_to_txt(self):
        """write the emd to the txt file"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            node_embed = self.sess.run(modes[i].node_embed)
            a = np.array(range(self.n_node)).reshape(-1, 1)
            node_embed = np.hstack([a, node_embed])
            node_embed_list = node_embed.tolist()
            node_embed_str = ["\t".join([str(x) for x in line]) + "\n" for line in node_embed_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(config.n_node) + "\t" + str(config.n_embed) + "\n"] + node_embed_str
                f.writelines(lines)


    def eval_test(self):
        """do the evaluation when training

        :return:
        """
        results = []
        if config.app == "link_prediction":
            for i in range(2):
                LPE = elp.LinkPredictEval(config.emb_filenames[i], config.test_filename, config.test_neg_filename, config.n_node, config.n_embed)
                result = LPE.eval_link_prediction()
                results.append(config.modes[i] + ":" + str(result) + "\n")


        with open(config.result_filename, mode="a+") as f:
            f.writelines(results)

if __name__ == "__main__": 
    g_g = GraphGan()
    g_g.train_gan()
