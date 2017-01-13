import tensorflow as tf

class rbm:
    def __init__(self, shape, para, data):
        self.para = para
        self.sess = tf.Session()
        self.data = data
        self.W = tf.Variable(tf.random_normal(shape[0], shape[1]), name = "W")
        self.a = tf.Variable(tf.zeros(shape[0]), name = "a")
        self.b = tf.Variable(tf.zeros(shape[1])), name = "b")
        self.V = tf.placeholder("float", [None, shape[0]])
        init_op = tf.global_
        print "rbm init completely"
        pass
    def doTrain(self):
        pass
    def cdK(self)
