import numpy as np
from utils.utils import *
import random
import copy
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

class Graph(object):
    def __init__(self, file_path, ng_sample_ratio):
        suffix = file_path.split('.')[-1]
        self.st = 0
        self.is_epoch_end = False
        if suffix == "txt":
            fin = open(file_path, "r")
            firstLine = fin.readline().strip().split()
            self.N = int(firstLine[0])
            self.E = int(firstLine[1])
            self.__is_epoch_end = False
            self.adj_matrix = dok_matrix((self.N, self.N), np.int_)
            self.links = np.zeros([self.E + int(ng_sample_ratio*self.N) , 3], np.int_)
            count = 0
            for line in fin.readlines():
                line = line.strip().split()
                self.adj_matrix[int(line[0]),int(line[1])] += 1
                self.adj_matrix[int(line[1]),int(line[0])] += 1
                self.links[count][0] = int(line[0])
                self.links[count][1] = int(line[1])
                self.links[count][2] = 1
                count += 1
            fin.close()
            if (ng_sample_ratio > 0):
                self.__negativeSample(int(ng_sample_ratio*self.N), count, self.adj_matrix.copy())
            self.order = np.arange(self.N)
            self.adj_matrix = self.adj_matrix.tocsr()
            print "getData done"
            print "Vertexes : %d  Edges : %d ngSampleRatio: %f" % (self.N, self.E, ng_sample_ratio)
        else:
            pass
            #TODO read a mat file or something like that.
        
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
        
    def load_label_data(self, filename):
        with open(filename,"r") as fin:
            firstLine = fin.readline().strip().split()
            self.label = np.zeros([self.N, int(firstLine[1])], np.bool)
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(' : ')
                if len(line) > 1:
                    labels = line[1].split()
                    for label in labels:
                        self.label[int(line[0])][int(label)] = True

    
    def sample(self, batch_size, do_shuffle = True, with_label = False):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.order[0:self.N])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False 
        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.order[self.st:en]     
        mini_batch.X = self.adj_matrix[index].toarray()
        mini_batch.adjacent_matriX = self.adj_matrix[index].toarray()[:][:,index]
        if with_label:
            mini_batch.label = self.label[index]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch
    
    def subgraph(self, method, sample_ratio):
        new_N = int(sample_ratio * self.N)
        cur_N = 0
        if method == 'link':
            new_links = []
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            while (cur_N < new_N):
                p = int(random.random() * self.E)
                link = self.links[p]
                if self.adj_matrix[link[0]][link[1]] == 0:
                    new_links.append(link)
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    if link[0] not in self.order:
                        self.order[link[0]] = 1
                        cur_N += 1
                    if link[1] not in self.order:
                        self.order[link[1]] = 1
                        cur_N += 1
            self.links = new_links
            self.order = self.order.keys()
            self.N = new_N
            print len(self.links)
            return self
        elif method == "node":
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                p = int(random.random() * self.N)
                if p not in self.order:
                    self.order[p] = 1
                    cur_N += 1
            for link in self.links:
                if link[0] in self.order and link[1] in self.order:
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    new_links.append(link)
            self.order = self.order.keys()
            self.N = new_N
            self.links = new_links
            print len(self.links)
            return self
            pass
        elif method == "explore": 
            new_adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                p = int(random.random() * self.N)
                k = int(random.random() * 100)
                for i in range(k):
                    if p not in self.order:
                        self.order[p] = 1
                        cur_N += 1
                    b = self.adj_matrix[p].nonzero()
                    b = b[0]
                    w = int(random.random() * len(b))
                    new_adj_matrix[p][b[w]] = 1
                    new_adj_matrix[b[w]][p] = 1
                    new_links.append([p,b[w],1])
                    p = b[w]
            self.order = self.order.keys()
            self.adj_matrix = new_adj_matrix
            self.N = new_N
            self.links = new_links
            print len(self.links)
            return self
            pass
    
