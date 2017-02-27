from AutoE import *
from utils import *

def setPara():
    para = {}
    para["learningRate"] = 0.01
    para["trainingEpochs"] = 20
    para["batchSize"] = 16
    para["beta"] = 10
    para["alpha"] = 1
    para['v'] = 0.0001
    para["dbn_init"] = False
    para["sparse_dot"] = True
    return para
    
if __name__ == "__main__":
    dataSet = "ca-Grqc.txt"
    data = getData(dataSet)
    para = setPara()
    para["M"] = data["N"]
    myAE = AutoE([data["N"],200,100], para, data)
    
    myAE.doTrain()
    embedding = myAE.getEmbedding(data["feature"])
    precisionK = getPrecisionK(embedding, data)
    
    print precisionK[2000]