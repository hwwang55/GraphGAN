import tensorflow as tf

class rbm:
    def __init__(self, shape, para, data):
        # shape[0] means the number of visible units
        # shape[1] means the number of hidden units
        self.para = para
        self.sess = tf.Session()
        self.data = datai
        stddev = 1.0 / np.sqrt(shape[0])
        self.W = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev = stddev), name = "W")
        self.bv = tf.Variable(tf.zeros(shape[0]), name = "a")
        self.bh = tf.Variable(tf.zeros(shape[1])), name = "b")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.buildModel()
        print "rbm init completely"
        pass
    def buildModel(self):
        self.hidden_holder = tf.placeholder("float", [None, shape[1]])
        self.visual_holder = tf.placeholder("float", [None, shape[0]])
        self.hidden = tf.nn.sigmoid(self.visual_holder * self.W + self.bh)
        self.visual = tf.nn.sigmoid(self.hidden_holder * tf.transpose(self.W) + self.bv)
        self.random = tf.Variable()
    def doTrain(self):
        for epoch in range(self.para["epoch"]):
            np.random.shuffle(self.data)
            for i in range(0, len(self.data), self.para["batch_size"]):
                X = self.data[i:i + self.para["batch_size"]]
                self.sess.run(upt, feed_dict = {self.visual_holder : X})
        pass
    def sample(probs):
        return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    def gibbs_sample(k);
        
    def cdK(self)
