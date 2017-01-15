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