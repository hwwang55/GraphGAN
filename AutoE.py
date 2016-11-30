import numpy as np
import tensorflow as tf


def preprocess(fileName):
	fin = open(fileName, "r")
	print "preprocessing...."
	firstLine = fin.readline().strip().split(" ")
	N = int(firstLine[0])
	E = int(firstLine[1])
	print N, E
	edge = np.zeros([N, N], np.int8)
	for line in fin.readlines():
		line = line.strip().split(' ')
		edge[int(line[0]),int(line[1])] += 1
	return {"N":N, "E":E, "feature":edge}

def setPara():
	para = {}
	para["learningRate"] = 0.01
	para["trainingEpochs"] = 20
	para["batchSize"] = 256
	para["displayStep"] = 1
	para["examplesToShow"] = 10
	para["n_hidden_1"] = 256
	return para

def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]), biases["encoder_b1"]))
	return layer_1

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder_h1"]), biases["decoder_b1"]))
	return layer_1

def doTrain(para, data):
	init = tf.initialize_all_variables()

	with tf.Session() as sess:
		sess.run(init)
		total_batch = int(data["N"] / para.batch_size)
		for epoch in range(training_epochs):
			for i in range(total_batch):
				batch_xs = data["feature"][i * para.batch_size: (i + 1) * para.batch_size] 
				_, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs})
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c))
		print("Optimization Finished!")

		#encoder_decode = sess.run(y_pred, feed_dict = {X: })


dataSet = "ca-Grqc.txt"

if __name__ == "__main__":
	data = preprocess(dataSet)
	para = setPara()
	# make network
	n_input = data["N"]
	n_hidden_1 = para["n_hidden_1"]
	X = tf.placeholder("float", [None, n_input])
	weights = {
		"encoder_h1" : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		"decoder_h1" : tf.Variable(tf.random_normal([n_hidden_1], n_input))
	}
	biases = {
		"encoder_b1" : tf.Variable(tf.random_normal([n_hidden_1])),
		"decoder_b1" : tf.Variable(tf.random_normal([n_input])),
	}
	encoderOP = encoder(X)
	decoderOP = decoder(encoderOP)

	y_pred = decoderOP
	y_true = X

	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.RMSProOptimizer(para.learning_rate).minimize(cost)

	dotrain(para, data)
