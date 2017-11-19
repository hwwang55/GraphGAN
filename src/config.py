n_embed = 20  # node embedding dimension
batch_size_dis = 128  # batch size for discriminator
batch_size_gen = 128  # batch size for generator
lambda_dis = 1e-5  # l2 loss regulation factor for discriminator
lambda_gen = 1e-5  # l2 loss regulation factor for generator
n_sample_dis = 20  # sample num for generator
n_sample_gen = 20  # sample num for discriminator
lr_dis = 1e-4  # learning rate for discriminator
lr_gen = 1e-3  # learning rate for discriminator
update_ratio = 1    # updating ratio when choose the trees
pretrain_emd_filename_d = "../pre_train/link_prediction/1.0/CA-GrQc_deepwalk_iters_3.emb"
pretrain_emd_filename_g = "../pre_train/link_prediction/1.0/CA-GrQc_deepwalk_iters_1.emb"
max_epochs = 300  # outer loop number
max_epochs_gen = 30  # loop number for generator
max_epochs_dis = 30  # loop number for discriminator
save_steps = 2
gen_for_d_iters = 10  # iteration numbers for generate new data for discriminator
max_degree = 0  # the max node degree of the network
app = 1  # choose the dataset, 0:"CA-AStroPh", 1:"CA-GrQc", 2:"ratings", 3:"blogcatalog", 4:"POS"
model_log = "../log/iteration/"
update_mode = "pair"  # when update generator, "pair" only update the nodes along the path, "softmax": also update the neighbors of the nodes along the path
reward_factor = 2  # factor when compute the reward in discriminator
# GAN_mode: "dis": only using discriminator, "gen": only using generator, "gan": both
GAN_mode = "dis"
# sample_mode  for discriminator: "just_pos": only use the neighborhoods, "exclude_pos": exclude the postive examples when sampling, "theory": dont not exclude any examples according to the theory
sample_mode = "just_pos"
# reward_mode: "constant": use some postive constant, "path_reward": all the pairs along the path have the same reward as <head, tail>,
# reward_mode": "pair_reward": the pair along the path have the different reward
reward_mode = "constant"
reward = 1  # only used for reward_mode = 0
use_mul = False # control if use the multiprocessing when constructing trees
load_model = False  # if load the model for continual training
gen_update_iter = 200
window_size = 2
walk_mode = None
walk_length = None