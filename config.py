class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "./GraphData/ca-Grqc.txt"
        ## embedding data
        self.embedding_filename = "ca-Grac" 
        ## hyperparameter
        self.struct = [None, 500, 100]
        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 1
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 15
        
        ## para for training
        self.batch_size = 1024
        self.epochs_limit = 1000
        self.learning_rate = 0.001
        
        self.DBN_init = False
        self.sparse_dot = False
        self.sample_method = "Point"  # Edge\Point\Community
        self.ng_sample_ratio = 0 # negative sample ratio
        
        
