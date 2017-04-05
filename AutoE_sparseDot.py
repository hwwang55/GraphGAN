import numpy as np
import tensorflow as tf
import time
import copy
import random
from utils import getData, getPrecisionK
from rbm import *
import scipy.io as sio

class AutoE_sparseDot:
    def __init__(self, shape, para, data):
        self.layers = len(shape)
        self.para = para
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        #self.sess = tf.Session(config = config)
        self.sess = tf.Session()
        self.isInit = False
        self.data = data
        self.W = {}
        self.b = {}
        for i in range(len(shape) - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([shape[i], shape[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([shape[i+1]]), name = name)
        shape.reverse()
        for i in range(len(shape) - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([shape[i], shape[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([shape[i+1]]), name = name)
        shape.reverse()
        self.shape = shape
        # input
                
        self.weight = tf.placeholder("float", [None, None])
        if self.para["sparse_dot"]:
            self.X_sp_indices = tf.placeholder(tf.int64)
            self.X_sp_ids_val = tf.placeholder(tf.float32)
            self.X_sp_shape = tf.placeholder(tf.int64)
            self.X = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        self.X1 = tf.placeholder("float", [None, para["M"]])
        
        self.makeStructure()
        self.cost = self.makeCost()
        
        #optimizer
        #self.optimizer = tf.train.GradientDescentOptimizer(self.para["learningRate"]).minimize(self.cost)
        #self.optimizer = tf.train.AdadeltaOptimizer(self.para["learningRate"]).minimize(self.cost)
        #self.optimizer = tf.train.AdagradOptimizer(self.para["learningRate"]).minimize(self.cost)
        
        #need another parameter
        #self.optimizer = tf.train.AdagradDAOptimizer(self.para["learningRate"]).minimize(self.cost) 
        #self.optimizer = tf.train.MomentumOptimizer(self.para["learningRate"]).minimize(self.cost)
        
        #self.optimizer = tf.train.AdamOptimizer(self.para["learningRate"]).minimize(self.cost)
        #self.optimizer = tf.train.FtrlOptimizer(self.para["learningRate"]).minimize(self.cost)
        #self.optimizer = tf.train.ProximalGradientDescentOptimizer(self.para["learningRate"]).minimize(self.cost)
        #self.optimizer = tf.train.ProximalAdagradOptimizer(self.para["learningRate"]).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(self.para["learningRate"]).minimize(self.cost)

    
    def makeStructure(self):
        #network structure
        if self.para["sparse_dot"]:
            self.encoderOP1 = self.encoder(self.X)
        else:
            self.encoderOP1 = self.encoder(self.X1)
        self.decoderOP1 = self.decoder(self.encoderOP1)
    
    def encoder(self, x):
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            if self.para["sparse_dot"] and i == 0:
                #another way to do sparseDot
                #x = tf.nn.sigmoid(tf.matmul(x, self.W[name], a_is_sparse = True) + self.b[name])
                x = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(x, self.W[name]) + self.b[name])
            else:
                x = tf.nn.sigmoid(tf.matmul(x, self.W[name]) + self.b[name])
        return x

    def decoder(self, x):
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            x = tf.nn.sigmoid(tf.matmul(x, self.W[name]) + self.b[name])
        return x
        
    def makeCost(self):
        #cost function
        self.cost2nd = self.get2ndCost(self.X1, self.decoderOP1)
        self.cost1st = self.get1stCost(self.encoderOP1, self.weight)
        self.costReg = self.getRegCost(self.W, self.b)
        
        #deleting the 2ndcost get no speeding up
        #return self.para['gamma'] * self.cost1st + self.para['v'] * self.costReg
        return self.para['gamma'] * self.cost1st + self.para['alpha'] * self.cost2nd + self.para['v'] * self.costReg
    
    def get1stCost(self, Y1, weight):
        D = tf.diag(tf.reduce_sum(weight,1))
        L = D - weight
        return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(Y1),L),Y1))

    def get2ndCost(self, X, newX):
        B = X * (self.para['beta'] - 1) + 1
        return tf.reduce_sum(tf.pow((newX - X)* B, 2))

    def getRegCost(self, weight, biases):
        ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
        ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
        return ret

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        self.isInit = True
 
    
    def displayResult(self, epoch, stTime):
        print "Epoch:", '%04d' % (epoch),
        costTotal = self.getCost()
        print "cost=", costTotal, 
        print "time : %.3fs" % (time.time() - stTime)
    
    def doInit(self): 
        init = tf.global_variables_initializer()        
        self.sess.run(init)
        if self.para["dbn_init"]:
            data = copy.copy(self.data["feature"])
            shape = self.shape
            for i in range(len(shape) - 1):
                myRBM = rbm([shape[i], shape[i+1]], {"epoch":0, "batch_size": 64, "learning_rate":0.1}, data)
                myRBM.doTrain()
                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                self.assign(self.W[name], W)
                self.assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                self.assign(self.W[name], W.transpose())
                self.assign(self.b[name], bv)
                data = myRBM.getH(data)
        self.cost = 0
        self.countCvg = 0
        self.isInit = True
    
    def checkStop(self, cur_cost):
        if (self.cost == 0):
            self.cost = min(cur_cost, self.cost)
            return False
        if (abs(self.cost - cur_cost) < 0.01 * self.cost):
            self.countCvg += 1
            if (self.countCvg > 3):
                return True
            else:
                return False
        self.cost = min(cur_cost, self.cost)
        return False

    def assign(self, a, b):
        op = a.assign(b)
        self.sess.run(op)
        
    def doTrain(self):
        para = self.para
        data = self.data
        if (not self.isInit):
            self.doInit()
        total_batch = int(data["N"] / para["batchSize"])
        print "Total_batch:", total_batch
        initT = time.time()
        totalEpoch = para["trainingEpochs"]
        epoch = 0
        while (epoch < totalEpoch):
            order = np.arange(data["N"])
            feed_dict = {}
            np.random.shuffle(order)
            all_time = 0 
            for i in range(total_batch):
                st = i * para["batchSize"]
                en =(i+1) * para["batchSize"]
                index = order[st:en]
                batchX1 = data["feature"][index]
                weight = data["feature"][index][:,index]
                if self.para["sparse_dot"]:
                    x_ind = np.vstack(np.where(batchX1)).astype(np.int64).T
                    x_shape = np.array(batchX1.shape).astype(np.int64)
                    x_val = batchX1[np.where(batchX1)]
                    feed_dict = {self.X_sp_indices: x_ind, self.X_sp_shape:x_shape, self.X_sp_ids_val: x_val, self.weight : weight}
                else:
                    feed_dict = {self.X1: batchX1, self.weight: weight}
                stT = time.time()
                _ = self.sess.run(self.optimizer, feed_dict = feed_dict)
                all_time = all_time + time.time() - stT
                #print "mini batch %d time : %.3fs" % (i, all_time)
            cur_Cost = self.sess.run(self.cost, feed_dict = feed_dict)
            print "epoch: %d time : %.3fs, cost: %.3f" % (epoch, time.time() - initT, cur_cost)
            if (self.checkStop(cur_cost)):
                break
        print "Optimization Finished!"
    
    def getEmbedding(self, data):
        return  self.sess.run(self.encoderOP1, feed_dict = {self.X1: data})

    def getW(self):
        return self.sess.run(self.W)
    def getB(self):
        return self.sess.run(self.b)
    def close(self):
        self.sess.close()

    

