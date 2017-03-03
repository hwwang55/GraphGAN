from AutoE import *
from AutoE_sparseDot import *
from utils import *


def setPara():
    para = {}
    para["learningRate"] = 0.001
    para["trainingEpochs"] = 1000
    para["batchSize"] = 1000
    para["beta"] = 15
    para["alpha"] = 1
    para["gamma"] = 5
    para['v'] = 0.1
    para["dbn_init"] = False
    para["sparse_dot"] = False
    return para

dataSet = "../NetworkData/blogCatalog3.txt"

data = getData(dataSet)
para = setPara()
para["M"] = data["N"]
myAE = AutoE_sparseDot([data["N"],1000,100], para, data)    

if __name__ == "__main__":
    myAE.doTrain()
    embedding = myAE.getEmbedding(data["feature"])
    sio.savemat('embedding.mat',{'embedding':embedding})
