"""the generator class
use the model to initialize and add some other structures different from the discriminator
"""
import tensorflow as tf
import config


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.variable_scope('Generator'):
            self.node_embed =  tf.get_variable(name="node_embed", shape=self.node_emd_init.shape,
                                               initializer=tf.constant_initializer(self.node_emd_init), trainable=True)
            self.node_b = tf.Variable(tf.zeros([self.n_node]))
        self.all_score = tf.matmul(self.node_embed, self.node_embed, transpose_b=True) + self.node_b
        # placeholder
        self.q_node = tf.placeholder(tf.int32, shape=[None])
        self.rel_node = tf.placeholder(tf.int32, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])
        # look up embeddings
        self.q_embedding= tf.nn.embedding_lookup(self.node_embed, self.q_node)  # batch_size*n_embed
        self.rel_embedding = tf.nn.embedding_lookup(self.node_embed, self.rel_node)  # batch_size*n_embed
        self.i_bias = tf.gather(self.node_b, self.rel_node)
        score = tf.reduce_sum(self.q_embedding*self.rel_embedding, axis=1) + self.i_bias
        i_prob = tf.nn.sigmoid(score)
        # clip value
        self.i_prob = tf.clip_by_value(i_prob, 1e-5, 1)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) \
                        + config.lambda_gen* (tf.nn.l2_loss(self.rel_embedding) + tf.nn.l2_loss(self.q_embedding))
        g_opt = tf.train.AdamOptimizer(config.lr_gen)
        self.gan_updates = g_opt.minimize(self.gan_loss)

