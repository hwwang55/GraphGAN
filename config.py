class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "GraphData/blogCatalog3.txt"
        self.label_file_path = "GraphData/blogCatalog3-groups.txt"
        ## embedding data
        self.embedding_filename = "embeddingResult/blogCatalog3" 
        ## hyperparameter
        self.struct = [None, 1000, 100]
        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 1000
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 90
        
        ## para for training
        #self.rN = 0.9
        self.batch_size = 1000
        self.epochs_limit = 1000
        self.learning_rate = 0.001
        
        self.DBN_init = False
        self.sparse_dot = False
        self.ng_sample_ratio = 0.0 # negative sample ratio
        
        
