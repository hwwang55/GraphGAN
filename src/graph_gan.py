import generator_pair
import generator
import discriminator
import utils
import collections
import tqdm
import copy
import numpy as np
import tensorflow as tf
import hickle
import time
import os
import multiprocessing
import config


class GraphGan(object):
    def __init__(self):
        """initialize the parameters, prepare the data and build the network"""
        random_state = np.random.randint(0, 100000)
        self.app = config.app
        self.datasets = {0: "CA-AstroPh", 1: "CA-GrQc", 2: "ratings", 3: "blogcatalog", 4: "POS", 5: "brazil"}
        if self.app in [0, 1]:
            # the graph information, list whose elment is also a list like a pair nodes [0, 2]
            self.train_filename = "../data/link_prediction/" + self.datasets[self.app] + "_undirected_train.txt"
            self.test_filename = "../data/link_prediction/" + self.datasets[self.app] + "_test.txt"
            self.n_node, self.linked_nodes = utils.read_edges(self.train_filename, self.test_filename)
            self.root_nodes = [i for i in range(self.n_node)]
        if self.app == 2:
            # the graph information, list whose elment is also a list like a pair nodes [0, 2]
            self.train_filename = "../data/" + self.datasets[self.app] + "_undirected_train.txt"
            self.test_filename = "../data/" + self.datasets[self.app] + "_test.txt"
            self.user_num, self.item_num, self.linked_nodes = utils.read_edges(self.train_filename, self.test_filename, mode="recommend")
            self.n_node = self.user_num + self.item_num
            self.user_nodes = [i for i in range(self.user_num)]
            self.item_node = [i for i in range(self.user_num, self.user_num + self.item_num)]
            self.nodes = [i for i in range(self.n_node)]
            self.root_nodes = self.user_nodes
        if self.app in [3, 4]:
            # the graph information, list whose elment is also a list like a pair nodes [0, 2]
            self.train_filename = "../data/" + self.datasets[self.app] + "_undirected_train.txt"
            self.test_filename = "../data/" + self.datasets[self.app] + "_test.txt"
            self.n_node, self.linked_nodes = utils.read_edges(self.train_filename, self.test_filename)
            self.root_nodes = [i for i in range(self.n_node)]
        if self.app == 5:  # for testing
            self.train_filename = "../data/visualization/brazil-fligths.txt"
            self.test_filename = ""
            self.n_node, self.linked_nodes = utils.read_edges(self.train_filename, self.test_filename)
            self.root_nodes = [i for i in range(self.n_node)]
        self.emb_filename_dis = "../pre_train/GraphGAN/" + self.datasets[self.app] + "_" + str(config.update_mode) + "_"\
                                + str(config.GAN_mode) + "_" + str(config.sample_mode) + "_" + str(config.reward_mode) + "_" \
                                + str(config.window_size) + "_" + str(config.walk_mode) + "_" + str(config.walk_length) + "_" + str(random_state) + "_dis"
        self.emb_filename_gen = "../pre_train/GraphGAN/" + self.datasets[self.app] + "_" + str(config.update_mode) + "_" + str(config.GAN_mode) + "_" + str(config.sample_mode) + "_" \
                                + str(config.reward_mode) + "_" + str(config.window_size) + "_" + str(config.walk_mode) + "_" + str(config.walk_length) + "_" + str(random_state) + "_gen"
        self.result_filename = {"dis": self.datasets[self.app] + "_" + str(config.update_mode) + "_" + str(config.GAN_mode) + "_" + str(config.sample_mode) + "_" + str(config.reward_mode) + "_" + str(config.window_size) + "_" + str(config.walk_mode) + "_" + str(config.walk_length) + "_" + str(random_state) + "_dis.txt",
                                "gen": self.datasets[self.app] + "_" + str(config.update_mode) + "_" + str(config.GAN_mode) + "_" + str(config.sample_mode) + "_" + str(config.reward_mode) + "_" + str(config.window_size) + "_" + str(config.walk_mode) + "_" + str(config.walk_length) + "_" + str(random_state) + "_gen.txt"}
        self.discriminator = None
        self.generator = None
        print("start reading initial embeddings")
        # read the initial embeddings
        self.node_embed_init_d = utils.read_emd(filename=config.pretrain_emd_filename_d, n_node=self.n_node, n_embed=config.n_embed)
        self.node_embed_init_g = utils.read_emd(filename=config.pretrain_emd_filename_g, n_node=self.n_node, n_embed=config.n_embed)
        print("finish reading initial embeddings")
        # use the BFS to construct the trees
        trees_filename = "../data/" + str(self.app) + "_trees.hkl"
        if os.path.exists(trees_filename):
            print("Load the trees")
            f = open(trees_filename, "r")
            self.trees = hickle.load(f)
            f.close()
        else:
            # use the BFS to construct the trees
            print("Constructing Trees")
            # f = open(trees_filename, "w")
            if self.app in [0, 1, 3, 4, 5]:  # link prediction or classification
                self.mul_construct_trees(self.root_nodes)
            if self.app in [2]:  # classification
                self.mul_construct_trees_for_recommend(self.user_nodes)
            # hickle.dump(self.trees, f)
            # f.close()
        config.max_degree = utils.get_max_degree(self.linked_nodes)
        self.build_gan()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

    def sample_for_gan(self, root, tree, sample_num, all_score, sample_for_dis):
        if config.walk_mode == "random_walk":
            return self.sample_for_gan_rw(root, tree, sample_num, all_score, sample_for_dis)
        elif config.walk_mode == "back":
            return self.sample_for_gan_back(root, tree, sample_num, all_score, sample_for_dis)

    def sample_for_gan_rw(self, root, tree, sample_num, all_score, sample_for_dis):
        """ sample the nodes from the tree

        Args:
            root: int, root, the query
            tree: dict, tree information
            sample_num: the number of the desired sampling nodes
        Returns:
            sample: list, include the index of the sampling nodes  ##
        """

        assert sample_for_dis in [False, True]

        sample = []
        self.trace = []
        n = 0
        walk_length = 0

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
                # remove node_father
                if node_father in node_neighbor:
                    node_neighbor.remove(node_father)
                if len(node_neighbor) == 0:  # if reach the leaf, stop it
                    break
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
                walk_length = walk_length + 1
                if walk_length == config.walk_length:  # if the walk length reaches the threshold
                    break
            n = n + 1
        return sample


    def sample_for_gan_back(self, root, tree, sample_num, all_score, sample_for_dis):
        """ sample the nodes from the tree

        Args:
            root: int, root, the query
            tree: dict, tree information
            sample_num: the number of the desired sampling nodes
        Returns:
            sample: list, include the index of the sampling nodes  ##
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
                if config.sample_mode == "exclude_pos" and sample_for_dis == True:  # only sample the negative examples for discriminator, thus should exclude the root node tobe sampled
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
            # cores = multiprocessing.cpu_count() // 2
            cores = 4
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

    def mul_construct_trees_for_recommend(self, nodes):
        """multiprocessing to speed the processes of constructing trees

        Args:
            nodes:list, user nodes
        Returns:

        """

        if config.use_mul:
            t1 = time.time()
            cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cores)
            new_nodes = []
            node_per_core = self.user_num // cores
            for i in range(cores):
                if i != cores - 1:
                    new_nodes.append(nodes[i*node_per_core:(i+1)*node_per_core])
                else:
                    new_nodes.append(nodes[i*node_per_core:])

            trees_result = pool.map(self.construct_tree_for_recommend, new_nodes)
            self.trees = {}
            for tree in trees_result:
                self.trees.update(tree)
            t2 = time.time()
            print(t2 - t1)
        else:
            self.trees = self.construct_tree_for_recommend(nodes)
        #  serialized the trees to the disk
        print("Dump the trees to the disk")

    def construct_tree_for_recommend(self, user_ids):
        """construct the tree for the recommendation system

        the root of the tree is the user, all other nodes are the items
        """

        linked_nodes = self.linked_nodes
        trees = {}
        for root in tqdm.tqdm(user_ids):
            trees[root] = {}
            tmp = copy.copy(linked_nodes[root])
            trees[root][root] = [root] + tmp
            if len(tmp) == 0:  # isolated user
                continue

            queue = collections.deque(tmp)  # the nodes in this queue all are items
            for x in tmp:
                trees[root][x] = [root]
            used_items = set(tmp)

            while len(queue) > 0:
                cur_item = queue.pop()
                used_items.add(cur_item)
                for sub_user in linked_nodes[cur_item]:
                    for sub_item in linked_nodes[sub_user]:
                        if sub_item not in used_items:
                            queue.appendleft(sub_item)
                            used_items.add(sub_item)
                            trees[root][cur_item].append(sub_item)
                            trees[root][sub_item] = [cur_item]

        return trees

    def build_discriminator(self):
        """initialize the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def build_generator(self):
        """initialize the generator"""

        with tf.variable_scope("generator"):
            if config.update_mode == "pair":
                self.generator = generator_pair.Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)
            else:
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
                if config.sample_mode in ["exclude_pos", "theory"]:  # "exclude_pos":use the True negative examples from the generator, "theory": use normal generator
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
        if config.update_mode == "softmax":
            self.train_gan_for_softmax()
        elif config.update_mode == "pair":
            self.train_gan_for_pair()

    def train_gan_for_pair(self):
        """train the whole graph gan network"""

        assert config.update_mode in ["pair", "softmax"]
        assert config.sample_mode in ["just_pos", "exclude_pos", "theory"]
        assert config.GAN_mode in ["dis", "gen", "gan"]
        assert config.reward_mode in ["path_reward", "pair_reward", "constant"]
        assert config.walk_mode in ["back", "random_walk"]
        #  restore the model
        #  check if there exists checkpoint, if true, load it
        ckpt = tf.train.get_checkpoint_state(config.model_log)
        if ckpt and ckpt.model_checkpoint_path and config.load_model:
            print("Load the checkpoint: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            #  save the model
            if epoch % config.save_steps == 0 and epoch > 0:
                self.saver.save(self.sess, config.model_log + "model.ckpt")

            if config.GAN_mode in ["dis", "gan"]:  # "dis": only use discriminator, "gan": use both discriminator and generator
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

                    node_embed = self.sess.run(self.discriminator.node_embed)
                    filename = self.emb_filename_dis + "_" + str(epoch * config.max_epochs_dis + d_epoch) + ".txt"
                    self.save_emb(node_embed, filename)

            if config.GAN_mode in ["gen", "gan"]:
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
                                reward_gen = []
                                for ii in range(len(pairs)):
                                    path_pairs = pairs[ii]
                                    for pair in path_pairs:
                                        q_node_gen.append(pair[0])
                                        rel_node_gen.append(pair[1])

                                if config.reward_mode == "path_reward":
                                    reward = self.sess.run(self.discriminator.reward,
                                                           {self.discriminator.q_node: np.array(root_nodes_gen),
                                                            self.discriminator.rel_node: np.array(rel_nodes)})
                                    # extend the reward corresponding the pairs
                                    reward_gen = []
                                    for ii in range(len(pairs)):
                                        for jj in range(len(pairs[ii])):
                                            reward_gen.append(reward[ii])
                                elif config.reward_mode == "constant":
                                    # extend the reward corresponding the pairs
                                    reward_gen = len(q_node_gen)*[config.reward]
                                elif config.reward_mode == "pair_reward":
                                    reward_gen, node_embed = self.sess.run([self.discriminator.score, self.discriminator.node_embed],
                                                               {self.discriminator.q_node: np.array(q_node_gen),
                                                                self.discriminator.rel_node: np.array(rel_node_gen)})
                                    # print(reward_gen)

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
                    node_embed = self.sess.run(self.generator.node_embed)
                    filename = self.emb_filename_gen + "_" + str(epoch * config.max_epochs_gen + g_epoch) + ".txt"
                    self.save_emb(node_embed, filename)
            print("Evaluation")
            self.write_emb_to_txt(epoch)
            self.eval_test(epoch)  # evaluation

    def train_gan_for_softmax(self):
        """train the whole graph gan network"""

        #  restore the model
        #  check if there exists checkpoint, if true, load it
        ckpt = tf.train.get_checkpoint_state(config.model_log)
        if ckpt and ckpt.model_checkpoint_path and config.load_model:
            print("Load the checkpoint: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        for epoch in tqdm.tqdm(range(config.max_epochs)):
            #  save the model
            if epoch % config.save_steps == 0 and epoch > 0:
                self.saver.save(self.sess, config.model_log + "model.ckpt")
            if config.GAN_mode in ["dis", "gan"]:  # "dis": only use discriminator, "gan": use both discriminator and generator
                for d_epoch in tqdm.tqdm(range(config.max_epochs_dis)):
                    if d_epoch % config.gen_for_d_iters == 0:  # every gen_for_d_iters round, we generate new data
                        self.generate_for_d()
                    train_size = len(self.samples_q)
                    #  traverse the whole training dataset sequentially, train the discriminator
                    index = 1
                    while True:
                        if index > train_size:
                            break
                        if index + config.batch_size_dis <= train_size + 1:
                            input_q_node, input_rel_node, input_label = self.get_batch_data(index, config.batch_size_dis)
                        else:
                            input_q_node, input_rel_node, input_label = self.get_batch_data(index, train_size - index + 1)
                        index += config.batch_size_dis
                        _, loss = self.sess.run([self.discriminator.d_updates, self.discriminator.pre_loss], {self.discriminator.q_node: np.array(input_q_node),\
                                          self.discriminator.rel_node: np.array(input_rel_node), self.discriminator.label: np.array(input_label)})

                    node_embed = self.sess.run(self.discriminator.node_embed)
                    filename = self.emb_filename_gen + "_" + str(epoch * config.max_epochs_dis + d_epoch) + ".txt"
                    self.save_emb(node_embed, filename)

            if config.GAN_mode in ["gen", "gan"]:
                for g_epoch in tqdm.tqdm(range(config.max_epochs_gen)):
                    cnt = 0
                    root_node_gen = []  # root node feeds into the network, same length as rel_node
                    rel_node = []  # the sample nodes of the root node
                    trace = []  # the trace when sampling the nodes, from the root to leaf  bach to leaf's father. e.g.: 0 - 1 - 2 -1
                    root_nodes = []  # just for record how many trees that have been traversed
                    all_score = self.sess.run(self.generator.all_score)  # compute the score for computing the probability when sampling nodes
                    loss_mean = []
                    for root_node in tqdm.tqdm(self.root_nodes, mininterval=3):  # random update trees
                        if np.random.rand() < config.update_ratio:
                            # sample the nodes according to our method.
                            # feed the reward from the discriminator and the sampled nodes to the generator.
                            # Update Generator
                            if cnt % config.gen_update_iter == 0 and cnt > 0:
                                if config.reward_mode == "path_reward":
                                    reward = self.sess.run(self.discriminator.reward,
                                                           {self.discriminator.q_node: np.array(root_node_gen),
                                                            self.discriminator.rel_node: np.array(rel_node)})
                                elif config.reward_mode == "constant":
                                    reward = len(rel_node)*[config.reward]
                                elif config.reward_mode == "pair_reward":
                                    q_node_gen = []
                                    rel_node_gen = []
                                    for ii in range(len(trace)):
                                        path = trace[ii]
                                        for jj in range(len(path)-1):
                                            q_node_gen.append(path[jj])
                                            rel_node_gen.append(path[jj+1])
                                    reward = self.sess.run(self.discriminator.reward,
                                                           {self.discriminator.q_node: np.array(q_node_gen),
                                                            self.discriminator.rel_node: np.array(rel_node_gen)})
                                if config.reward_mode == "pair_reward":
                                    is_pair = True
                                elif config.reward_mode in ["path_reward", "constant"]:
                                    is_pair = False
                                feed_dict = self.feed_dict(root_node_gen, reward, trace, is_pair)
                                _, loss, test_nn = self.sess.run([self.generator.gan_updates, self.generator.gan_loss, self.generator.i_prob],
                                                        feed_dict=feed_dict)
                                loss_mean.append(loss)
                                all_score = self.sess.run(self.generator.all_score)
                                root_node_gen = []
                                root_nodes = []
                                rel_node = []
                                trace = []
                                cnt = 0
                            sample = self.sample_for_gan(root_node, self.trees[root_node], config.n_sample_gen, all_score, sample_for_dis=False)
                            if len(sample) < config.n_sample_gen:
                                cnt = len(root_nodes)
                                continue
                            root_nodes.append(root_node)
                            root_node_gen.extend(len(sample)*[root_node])
                            rel_node.extend(sample)
                            trace.extend(self.trace)
                            cnt = cnt + 1
                    node_embed = self.sess.run(self.generator.node_embed)
                    filename = self.emb_filename_gen + "_" + str(epoch * config.max_epochs_gen + g_epoch) + ".txt"
                    self.save_emb(node_embed, filename)

    def feed_dict(self, root_node_gen, reward, trace, is_pair):
        """trace:dict <key, val>
        """
        q_node = []  # trace nodes
        rel_node = []  # neighbor nodes of the trace nodes
        node_position = []  # the trace nodes position in the neighbor nodes
        root_nodes = []
        prob_mask = []
        reward_feed = []
        trace_len = [0]
        ## be careful about this
        for i in range(config.n_sample_gen * config.gen_update_iter):
            q_node.extend(trace[i][:-1])
            root_nodes.extend((len(trace[i])-1)*[root_node_gen[i]])
            reward_feed.extend((len(trace[i])-1)*[reward[i]])
            trace_len.append(trace_len[-1]+len(trace[i][:-1]))

        if is_pair:
            reward_feed = reward
        for i in range(len(q_node)):
            neighbor = self.trees[root_nodes[i]][q_node[i]]
            if i in trace_len:
                neighbor = neighbor[1:]
            prob_mask.append(len(neighbor)*[0])
            prob_mask[-1].extend((config.max_degree-len(neighbor))*[-1e6])
            rel_node.append(self.padding_neighbor(neighbor))
            if i+1 in trace_len:
                node_position.append(neighbor.index(q_node[i-1]))
            else:
                # print(neighbor)
                # print(q_node[i+1])
                node_position.append(neighbor.index(q_node[i+1]))
        feeds = {}
        feeds[self.generator.q_node] = np.array(q_node).reshape([-1, 1])
        feeds[self.generator.rel_node] = np.array(rel_node)
        feeds[self.generator.node_position] = np.vstack([range(len(q_node)), node_position]).transpose()
        feeds[self.generator.reward] = np.array(reward_feed)
        feeds[self.generator.prob_mask] = np.array(prob_mask)
        batch_size = feeds[self.generator.q_node].shape[0]
        assert feeds[self.generator.rel_node].shape == (batch_size, config.max_degree)
        assert feeds[self.generator.node_position].shape == (batch_size, 2)
        assert feeds[self.generator.prob_mask].shape == (batch_size, config.max_degree)
        return feeds

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

    def write_emb_to_txt(self, epoch):
        """write the emd to the txt file"""

        flag = True
        for i in [self.generator, self.discriminator]:
            node_embed = self.sess.run(i.node_embed)
            a = np.array(range(self.n_node)).reshape(-1, 1)
            node_embed = np.hstack([a, node_embed])
            if flag:
                np.savetxt(self.emb_filename_gen + "_1_" + str(epoch) + ".txt", node_embed, fmt="%10.5f", delimiter='\t')
                flag = False
            else:
                np.savetxt(self.emb_filename_dis + "_1_" + str(epoch) + ".txt", node_embed, fmt="%10.5f", delimiter='\t')



import argparse
def parse_args():
    """
    parse the args
    :return:
    """
    parser = argparse.ArgumentParser(description="Run GraphGAN")
    parser.add_argument("--update_mode", type=str, default="pair")
    parser.add_argument("--GAN_mode", type=str, default="gan")
    parser.add_argument("--reward_mode", type=str, default="pair_reward")
    parser.add_argument("--sample_mode", type=str, default="theory")
    parser.add_argument("--window_size", type=int, default="2")
    parser.add_argument("--walk_mode", type=str, default="back")
    parser.add_argument("--walk_length", type=int, default="0")
    return parser.parse_args()


def main(args):
    config.GAN_mode = args.GAN_mode
    config.update_mode = args.update_mode
    config.reward_mode = args.reward_mode
    config.sample_mode = args.sample_mode
    config.window_size = args.window_size
    config.walk_mode = args.walk_mode
    config.walk_length = args.walk_length
    g_g = GraphGan()
    g_g.train_gan()
    g_g.write_emb_to_txt()

if __name__ == "__main__":
    args = parse_args()
    main(args)

