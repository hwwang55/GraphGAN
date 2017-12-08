import numpy as np
batch_size_dis = 64  # batch size for discriminator
batch_size_gen = 64  # batch size for generator
lambda_dis = 1e-5  # l2 loss regulation factor for discriminator
lambda_gen = 1e-5  # l2 loss regulation factor for generator
n_sample_dis = 20  # sample num for generator
n_sample_gen = 20  # sample num for discriminator
update_ratio = 1    # updating ratio when choose the trees
save_steps = 10

lr_dis = 1e-4  # learning rate for discriminator
lr_gen = 1e-3  # learning rate for discriminator

max_epochs = 20  # outer loop number
max_epochs_gen = 30  # loop number for generator
max_epochs_dis = 30  # loop number for discriminator

gen_for_d_iters = 10  # iteration numbers for generate new data for discriminator
max_degree = 0  # the max node degree of the network
model_log = "../log/iteration/"

use_mul = False # control if use the multiprocessing when constructing trees
load_model = False  # if load the model for continual training
gen_update_iter = 200
window_size = 3
random_state = np.random.randint(0, 100000)
app = "link_prediction"
train_filename = "../../data/" + app + "/others" + "/CA-GrQc_undirected_train.txt"
test_filename = "../../data/link_prediction/CA-GrQc_test.txt"
test_neg_filename = "../../data/link_prediction/CA-GrQc_test_neg.txt"
n_embed = 50
n_node = 5242
pretrain_emd_filename_d = "../../pre_train/" + app + "/CA-GrQc_pre_train.emb"
pretrain_emd_filename_g = "../../pre_train/" + app + "/CA-GrQc_pre_train.emb"
modes = ["dis", "gen"]
emb_filenames = ["../../pre_train/" + app + "/CA-GrQc_" + modes[0] + "_" + str(random_state) + ".emb",
                 "../../pre_train/" + app + "/CA-GrQc_" +  modes[1] + "_" + str(random_state) + ".emb"]
result_filename = "../../results/" + app + "/CA-GrQc_" +  str(random_state) + ".txt"

