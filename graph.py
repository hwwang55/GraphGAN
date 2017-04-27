import numpy as np
from utils import Dotdict


class Graph(object):
    def __init__(self, file_path, ng_sample_ratio):
        suffix = file_path.split('.')[-1]
        if suffix == "txt":
            fin = open(file_path, "r")
            firstLine = fin.readline().strip().split(" ")
            self.N = int(firstLine[0])
            self.E = int(firstLine[1])
            self.__is_epoch_end = False
            self.__adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.__edges = np.zeros([E + ng_sample_ratio*self.N,3], np.int_)
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
            if (ng_sample_ratio > 0):
                self.__negativeSample(ng_sample_ratio * self.N, count, edges.copy())
            self.__order = np.arange(self.N)
            print "getData done"
        else:
            pass
            #TODO read a mat file or something like that.
        print "Vertexes : %d  Edges : %d ngSampleRatio" % (self.N, E, ng_sample_ratio)
        
    def __negativeSample(self, ngSample, count, edges):
        print "negative Sampling"
        size = 0
        while (size < ngSample):
            xx = random.randint(0, self.N-1)
            yy = random.randint(0, self.N-1)
            if (xx == yy or edges[xx][yy] != 0):
                continue
            edges[xx][yy] = -1
            edges[yy][xx] = -1
            self.links[size + count] = [xx, yy, -1]
            size += 1
        print "negative Sampling done"
        
        
    def sample(self, batch_size):
        if self.__is_epoch_end:
            np.random.shuffle(self.__order)
            self.st = 0
        
        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.order[st:en]
        mini_batch.X = self.__adj_matrix[index]
        mini_batch.adjacent_matriX = self.__adj_matrix[index][:,index]
        self.st = en
        return mini_batch
    
    def is_Epoch_end():
        return self.__is_epoch_end
