'''
Reference implementation of SDNE

Author: Xuanrong Yao, Daixin Wang

for more detail, refer to the paper:
SDNE : structral deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

# -*- coding: utf-8 -*-
import argparse
from AutoE import *
from AutoE_sparseDot import *
from utils import *

def parse_args():
    '''
    Parses the SDNE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run SDNE.")

    parser.add_argument('--input', nargs='?', default='../Graph/ca-Grqc.txt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs. Default is 100')
    
    parser.add_argument('--learningRate', type=float, default=0.001,
                        help='learning rate. Default is 0.001.')
                        
    parser.add_argument('--batchSize', type=float, default=64,
                        help='mini-batch size. Default is 64.')
                        
    parser.add_argument('--beta', type=float, default = 15,
                        help='the weight balanced value')
                        
    parser.add_argument('--alpha', type=float, default = 1,
                        help="the weight of Loss_2nd")
    
    parser.add_argument('--gamma', type=float, default = 5,
                        help="the weight of Loss_1st")
    
    parser.add_argument('--reg', type=float, default = 5,
                        help="the weight of regularization term")

    parser.add_argument('--DBN_init', dest="dbn_init", action="store_true", default=False,
                        help="use the DBN to initialize the NN")
    
    parser.add_argument('--ngSampleRatio', dest="ngSampleRatio", default=0,
                        help="the negative sample ratio. Default is 0")
    
    parser.add_argument('--sparseDot', dest="sparse_dot", action="store_true", 
                        help="use sparseDot")
    
    return parser.parse_args()

if __name__ == "__main__":
    para = parse_args()
    data = getData(para.input, para.ngSampleRatio)
    para.M = data["N"]
    myAE = AutoE_sparseDot([para.M,1000, 100], para, data)    
    myAE.doTrain()
    embedding = myAE.getEmbedding(data["feature"])
    sio.savemat('embedding.mat',{'embedding':embedding})
