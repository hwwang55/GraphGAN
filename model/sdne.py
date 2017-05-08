import numpy as np
import tensorflow as tf
import time
import copy
import random

class SDNE:
    def __init__(self, config):
    
        self.is_variables_init = False
    
        ######### not running out gpu sources ##########
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config =  tf_config)

        ############ define variables ##################
        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse_dot = config.sparse_dot
        self.W = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()
        ###############################################
        ############## define input ###################
                
        self.adjacent_matriX = tf.placeholder("float", [None, None])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64)
        self.X_sp_ids_val = tf.placeholder(tf.float32)
        self.X_sp_shape = tf.placeholder(tf.int64)
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        #
        self.X = tf.placeholder("float", [None, config.struct[0]])
        
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
        

    
    def __make_compute_graph(self):
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        def encoder_sp(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                if i == 0:
                    X = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        if self.sparse_dot:
            self.H = encoder_sp(self.X_sp)
        else:
            self.H = encoder(self.X)
        self.X_reconstruct = decoder(self.H)
    

        
    def __make_loss(self, config):
        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            L = D - adj_mini_batch ## L is laplation-matriX
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return ret
            
        #Loss function
        self.loss_2nd = get_2nd_loss(self.X, self.X_reconstruct, config.beta)
        self.loss_1st = get_1st_loss(self.H, self.adjacent_matriX)
        
        # we don't need the regularizer term, since we have nagetive sampling.
        #self.loss_reg = get_reg_loss(self.W, self.b) 
        #return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg
        
        return config.gamma * self.loss_1st + config.alpha * self.loss_2nd

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        self.is_Init = True
    
    def do_variables_init(self, DBN_init = 0):
        def __assign(a, b):
            op = a.assign(b)
            self.sess.run(op)
        init = tf.global_variables_initializer()        
        self.sess.run(init)
        if DBN_init:
            pass
            ## TODO
            # data = copy.copy(self.data["feature"])
            # shape = self.shape
            # for i in range(len(shape) - 1):
                # myRBM = rbm([shape[i], shape[i+1]], {"epoch":0, "batch_size": 64, "learning_rate":0.1}, data)
                # myRBM.doTrain()
                # W, bv, bh = myRBM.getWb()
                # name = "encoder" + str(i)
                # self.assign(self.W[name], W)
                # self.assign(self.b[name], bh)
                # name = "decoder" + str(self.layers - i - 2)
                # self.assign(self.W[name], W.transpose())
                # self.assign(self.b[name], bv)
                # data = myRBM.getH(data)
        self.is_Init = True

    def __get_feed_dict(self, data):
        X = data.X
        if self.sparse_dot:
            X_ind = np.vstack(np.where(X)).astype(np.int64).T
            X_shape = np.array(X.shape).astype(np.int64)
            X_val = X[np.where(X)]
            return {self.X_sp_indices: X_ind, self.X_sp_shape:X_shape, self.X_sp_ids_val: X_val, self.adjacent_matriX : adjacent_matriX}
        else:
            return {self.X: data.X, self.adjacent_matriX: data.adjacent_matriX}
            
    def fit(self, data):
        if (not self.is_Init):
            print "Warning: the model isn't initialized, and will be initialized randomly"
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        _ = self.sess.run(self.optimizer, feed_dict = feed_dict)
    
    def get_loss(self, data):
        if (not self.is_Init):
            print "Warning: the model isn't initialized, and will be initialized randomly"
            self.do_variables_init()
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.H, feed_dict = self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)
        
    def get_B(self):
        return self.sess.run(self.b)
        
    def close(self):
        self.sess.close()

    

