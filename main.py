'''
Reference implementation of SDNE

Author: Xuanrong Yao, Daixin wang

for more detail, refer to the paper:
SDNE : structral deep network embedding
Wang, Daixin and Cui, Peng and Zhu, Wenwu
Knowledge Discovery and Data Mining (KDD), 2016
'''

#!/usr/bin/python2
# -*- coding: utf-8 -*-



from config import Config
from graph import Graph
from model.sdne import SDNE
from utils.utils import *
import scipy.io as sio
import time
import copy


if __name__ == "__main__":
    config = Config()
    
    graph_data = Graph(config.file_path, config.ng_sample_ratio)
    #graph_data.load_label_data(config.label_file_path)
    config.struct[0] = graph_data.N
    
    model = SDNE(config)    
    model.do_variables_init(graph_data, config.DBN_init)

    epochs = 0
    batch_n = 0
    
    origin_data = copy.deepcopy(graph_data)
    #graph_data = graph_data.subgraph(config.sample_method, config.sample_ratio)
    fout = open(config.embedding_filename + "-log.txt","w") 
    while (True):
        #graph_data.N = int(config.rN * graph_data.N)
        mini_batch = graph_data.sample(config.batch_size)
        loss = model.fit(mini_batch)
        batch_n += 1
        print "Epoch : %d, batch : %d, loss: %.3f" % (epochs, batch_n, loss)
        if graph_data.is_epoch_end:
            epochs += 1
            batch_n = 0
            print "Epoch : %d loss : %.3f" % (epochs, loss)
            if epochs % config.display == 0:
                embedding = None
                while (True):
                    mini_batch = graph_data.sample(config.batch_size, do_shuffle = False)
                    loss += model.get_loss(mini_batch)
                    if embedding is None:
                        embedding = model.get_embedding(mini_batch)
                    else:
                        embedding = np.vstack((embedding, model.get_embedding(mini_batch)))
                
                    if graph_data.is_epoch_end:
                        break

                result = check_link_reconstruction(embedding, graph_data, [20000,40000,60000,80000,100000])
                #data = origin_data.sample(origin_data.N, with_label = True)
                #check_multi_label_classification(model.get_embedding(data), data.label)
                print >> fout, epochs, result
                sio.savemat(config.embedding_filename + '-' + str(epochs) + '_embedding.mat',{'embedding':embedding})
            if epochs > config.epochs_limit:
                print "exceed epochs limit terminating"
                break
            last_loss = loss
        
    fout.close()
