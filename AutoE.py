import numpy as np
import tensorflow as tf
import time
import random

def negativeSample(ngSample, links, count, edges, N):
	size = 0
	while (size < ngSample):
		xx = random.randint(0, N-1)
		yy = random.randint(0, N-1)
		if (xx == yy or edges[xx][yy] != 0):
			continue
		edges[xx][yy] = -1
		edges[yy][xx] = -1
		links[size + count] = [xx, yy, -1]
		size += 1
	
def getData(fileName):
	fin = open(fileName, "r")
	print "preprocessing...."
	firstLine = fin.readline().strip().split(" ")
	N = int(firstLine[0])
	E = int(firstLine[1])
	print N, E
	ngSample = 0
	edges = np.zeros([N, N], np.int_)
	links = np.zeros([E + ngSample,3], np.int_)
	count = 0
	for line in fin.readlines():
		line = line.strip().split(' ')
		edges[int(line[0]),int(line[1])] += 1
		edges[int(line[1]),int(line[0])] += 1
		links[count][0] = int(line[0])
		links[count][1] = int(line[1])
		links[count][2] = 1
		count += 1
	fin.close()
	negativeSample(ngSample, links, count, edges.copy(), N)
	return {"N":N, "E":E, "feature":edges, "links": links}

class AutoE:
	def __init__(self, shape, para, data):
		self.layers = len(shape)
		self.para = para
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
		
		# input
		self.X1 = tf.placeholder("float", [None, para["M"]])
		self.X2 = tf.placeholder("float", [None, para["M"]])
		self.Sij = tf.placeholder("bool", [None])
		
		self.makeStructure()
		self.cost = self.makeCost()
		
		#optimizer
		self.optimizer = tf.train.RMSPropOptimizer(self.para["learningRate"]).minimize(self.cost)
	
	def makeStructure(self):
		#network structure
		self.encoderOP1 = self.encoder(self.X1)
		self.encoderOP2 = self.encoder(self.X2)

		self.decoderOP1 = self.decoder(self.encoderOP1)
		self.decoderOP2 = self.decoder(self.encoderOP2)
	
	def encoder(self, x):
		for i in range(self.layers - 1):
			name = "encoder" + str(i)
			x = tf.nn.sigmoid(tf.matmul(x, self.W[name]) + self.b[name])
		return x

	def decoder(self, x):
		for i in range(self.layers - 1):
			name = "decoder" + str(i)
			x = tf.nn.sigmoid(tf.matmul(x, self.W[name]) + self.b[name])
		return x
		
	def makeCost(self):
		#cost function
		self.cost2nd = self.get2ndCost(self.X1, self.decoderOP1) + self.get2ndCost(self.X2, self.decoderOP2)
		self.cost1st = self.get1stCost(self.encoderOP1, self.encoderOP2)
		self.costReg = self.getRegCost(self.W, self.b)
		return self.cost1st + self.para['alpha'] * self.cost2nd + self.para['v'] * self.costReg
	
	def get1stCost(self, Y1, Y2):
		return tf.reduce_sum(tf.pow(Y1 - Y2, 2))

	def get2ndCost(self, X, newX):
		B = X * (para['beta'] - 1) + 1
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

	def getCost(self):	
		return self.sess.run([self.cost1st, self.cost2nd, self.costReg], feed_dict = {self.X1 : self.data["feature"][data["links"][:,0]], self.X2 : self.data["feature"][data["links"][:,1]]})
	
	def displayResult(self, epoch, stTime):
		print "Epoch:", '%04d' % (epoch),
		costTotal = self.getCost()
		print "cost=", costTotal, 
		print "time : %.3fs" % (time.time() - stTime)
	
	def doInit(self): 
		init = tf.global_variables_initializer()		
		self.sess.run(init)
		self.isInit = True

	def doTrain(self):
		para = self.para
		data = self.data
		if (not self.isInit):
			self.doInit()
		total_batch = int(data["E"] / para["batchSize"])
		for epoch in range(para["trainingEpochs"]):
			np.random.shuffle(data["links"])
			stT = time.time()
			for i in range(total_batch):
				st = i * para["batchSize"]
				en =(i+1) * para["batchSize"]
				index = data["links"][st:en]
				batchX1 = data["feature"][index[:,0]]
				batchX2 = data["feature"][index[:,1]]
				_ = self.sess.run(self.optimizer, feed_dict = {self.X1:batchX1, self.X2:batchX2})
			self.displayResult(epoch, stT)
		print "Optimization Finished!"
	
	def getEmbedding(self, data):
		return  self.sess.run(self.encoderOP1, feed_dict = {self.X1: data})

	def getW(self):
		return self.sess.run(self.W)
	def getB(self):
		return self.sess.run(self.b)
	def close(self):
		self.sess.close()

def getSimilarity(result):
	print "getting similarity..."
	return np.dot(result, result.T)


def getPrecisionK(embedding, data):
	print "get precisionK..."
	similarity = getSimilarity(embedding).reshape(-1)
	sortedInd = np.argsort(similarity)
	cur = 0
	count = 0
	precisionK = []
	sortedInd = sortedInd[::-1]
	for ind in sortedInd[0:10000]:
		x = ind / data['N']
		y = ind % data['N']
		if (x == y):
			continue
		count += 1
		if (data["feature"][x][y] == 1):
			cur += 1 
		precisionK.append(1.0 * cur / count)
	return precisionK


def setPara():
	para = {}
	para["learningRate"] = 0.01
	para["trainingEpochs"] = 20
	para["batchSize"] = 64
	para["beta"] = 10
	para["alpha"] = 1
	para['v'] = 0.0001
	return para

dataSet = "ca-Grqc.txt"

data = getData(dataSet)
para = setPara()
para["M"] = data["N"]
myAE = AutoE([data["N"],500,100], para, data)	

if __name__ == "__main__":
	myAE.doTrain()
	embedding = myAE.getEmbedding(data["feature"])
	precisionK = getPrecisionK(embedding, data)
	
	print precisionK[2000]


