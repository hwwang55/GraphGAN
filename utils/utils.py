import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print "getting similarity..."
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print "get precisionK..."
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            if (x == y):
                continue
            count += 1
            if (data.adj_matrix[x][y] == 1):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    for index in check_index:
        print "precisonK[%d] %.2f" % (index, precisionK[index - 1])

def check_multi_label_classification(X, Y, test_ratio = 0.5):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.argsort(y_pred, axis = 1)[::-1]
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    def get_F1_score(Y, pred):
        tmp = Y & pred
        num_correct = tmp.sum(0)
        num_true = Y.sum(0)
        num_pred  = pred.sum(0)

        # the precision/recall/F1 for each label
        precision = 1.0 * num_correct / num_pred
        recall = 1.0 * num_correct / num_true
        F1 = (2.0 * num_correct) / (num_true+num_pred)

        # precision(num_correct==0)=0;
        # recall(num_correct==0)=0;
        # F1(num_correct==0)=0;

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_F1 = F1.mean()

        micro_precision = 1.0 * sum(num_correct) /  sum(num_pred)
        micro_recall = 1.0 * sum(num_correct) / sum(num_true)
        micro_F1 = 2.0 * sum(num_correct)/(sum(num_true) + sum(num_pred))
        
        return macro_F1, micro_F1
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    macro_f1, micro_f1 = get_F1_score(y_test, y_pred)
    print "macro_f1: %.4f" % (macro_f1)
    print "micro_f1: %.4f" % (micro_f1)
    #############################################


    