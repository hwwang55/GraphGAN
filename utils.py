import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def doClassification(X, Y, ratio):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = ratio)
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(y_train)
    print "macro-f1:", f1_score(y_test, y_pred, average = "macro")
    print "micro-f1:", f1_score(y_test, y_pred, average = "micro")


def negativeSample(ngSample, links, count, edges, N):
    print "negative Sampling"
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
    print "negative Sampling done"

def getData(fileName, ngSampleRatio):
    fin = open(fileName, "r")
    print "preprocessing...."
    firstLine = fin.readline().strip().split(" ")
    N = int(firstLine[0])
    E = int(firstLine[1])
    print N, E
    ngSample = int(ngSampleRatio * E)
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
    if (ngSample > 0):
        negativeSample(ngSample, links, count, edges.copy(), N)
    print "getData done"
    return {"N":N, "E":E, "feature":edges, "links": links}
	
def getSimilarity(result):
    print "getting similarity..."
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index)
    def get_precisionK(embedding, data, max_index):
        print "get precisionK..."
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd[0:max_index]:
            x = ind / data.N
            y = ind % data.N
            if (x == y):
                continue
            count += 1
            if (data.adj_matrix[x][y] == 1):
                cur += 1 
            precisionK.append(1.0 * cur / count)
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    for index in check_index:
        print "precisonK[%d] %.2f" % (index, precision[index])

    #############################################