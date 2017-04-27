class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "../Graph/ca-Grqc.txt"
        
        ## hyperparameter
        self.struct = [None, 1000, 100]
        ## the loss func is  // gamma * L1 + alpha * L2 + nv * regularTerm // 
        self.alpha = 5
        self.gamma = 5
        self.regular = 5
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 1
        
        ## para for training
        self.batch_size = 1000
        self.epochs_limit = 1000
        self.learning_rate = 0.001
        
        self.DBN_init = False
        self.sample_method = "Point"  # Edge\Point\Community
        self.ng_sample_ratio = 0 # negative sample ratio
        
        
