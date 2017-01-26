import tensorflow as tf
import time

sess = tf.Session()
a = tf.Variable(tf.random_uniform([10000,10000]))
a1 = tf.cast(a > 0.8, tf.int32)
a2 = tf.cast(a > 0.9, tf.int32)

init = tf.global_variables_initializer()
sess.run(init)

tt = time.time()
b1 = tf.matmul(a1, a1, a_is_sparse = True)
sess.run(b1)
print "time sparse : %.3fs" % (time.time() - stTime)

tt = time.time()
b1 = tf.matmul(a1, a1, a_is_sparse = False)
sess.run(b1)
print "time sparse : %.3fs" % (time.time() - stTime)