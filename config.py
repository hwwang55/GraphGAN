class Config(object):
    def __init__(self):
        ## graph data
        self.file_path = "../Data_SDNE/Flickr2_0.7.txt"
        #self.file_path = "GraphData/ca-Grqc.txt"
        #self.label_file_path = "GraphData/blogCatalog3-groups.txt"
        ## embedding data
        self.embedding_filename = "embeddingResult/Flickr2-0.7" 
        ## hyperparameter
        self.struct = [None, 1000, 128]
        ## the loss func is  // gamma * L1 + alpha * L2 + reg * regularTerm // 
        self.alpha = 500
        self.gamma = 1
        self.reg = 1
        ## the weight balanced value to reconstruct non-zero element more.
        self.beta = 10
        
        ## para for training
        #self.rN = 0.9
        self.batch_size = 64
        self.epochs_limit = 10
        self.learning_rate = 0.01
        self.display = 1

        self.DBN_init = True
        self.dbn_epochs = 500
        self.dbn_batch_size = 64
        self.dbn_learning_rate = 0.1

        self.sparse_dot = False
        self.ng_sample_ratio = 0.0 # negative sample ratio
        
        #self.sample_ratio = 1
        #self.sample_method = "node"
