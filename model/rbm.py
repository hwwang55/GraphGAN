import tensorflow as tf
from utils import getData
import numpy as np
class rbm:
    def __init__(self, shape, para, data):
        # shape[0] means the number of visible units
        # shape[1] means the number of hidden units
        self.para = para
        self.sess = tf.Session()
        self.data = data
        stddev = 1.0 / np.sqrt(shape[0])
        self.W = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev = stddev), name = "W")
        self.bv = tf.Variable(tf.zeros(shape[0]), name = "a")
        self.bh = tf.Variable(tf.zeros(shape[1]), name = "b")
        self.v = tf.placeholder("float", [None, shape[0]])
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.buildModel()
        print "rbm init completely"
        pass
    def buildModel(self):
        self.h = self.sample(tf.sigmoid(tf.matmul(self.v, self.W) + self.bh))
        #gibbs_sample
        v_sample = self.sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.bv))
        h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample, self.W) + self.bh))
        lr = self.para["learning_rate"] / tf.to_float(self.para["batch_size"])
        W_adder = self.W.assign_add(lr  * (tf.matmul(tf.transpose(self.v), self.h) - tf.matmul(tf.transpose(v_sample), h_sample)))
        bv_adder = self.bv.assign_add(lr * tf.reduce_mean(self.v - v_sample, 0))
        bh_adder = self.bh.assign_add(lr * tf.reduce_mean(self.h - h_sample, 0))
        self.upt = [W_adder, bv_adder, bh_adder]
        self.error = tf.reduce_sum(tf.pow(self.v - v_sample, 2))
    def doTrain(self):
        for epoch in range(self.para["epoch"]):
            np.random.shuffle(self.data)
            for i in range(0, len(self.data), self.para["batch_size"]):
                X = self.data[i:i + self.para["batch_size"]]
                self.sess.run(self.upt, feed_dict = {self.v : X})
            error = self.sess.run(self.error, feed_dict = {self.v : self.data})
            print "epoch : ", epoch, ":", error
        pass
	def getH(self, data):
		return self.sess.run(self.h, feed_dict = {self.v : data})
    def sample(self, probs):
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    def getWb(self):
        return self.sess.run([self.W, self.bv, self.bh])
    def getH(self, data):
        return self.sess.run(self.h, feed_dict = {self.v : self.data})
    def close(self):
        return self.sess.close()

if __name__ == "__main__":
    dataSet = "ca-Grqc.txt"
    data = getData(dataSet)["feature"]
    myRBM = rbm([data.shape[0], 200], {"epoch":200, "batch_size": 64, "learning_rate":0.1}, data)
    myRBM.doTrain()
    W, bv, bh = myRBM.getWb()
