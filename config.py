#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 128  # embedding dimension
HIDDEN_DIM = 128  # hidden state dimension of lstm cell
SEQ_LENGTH = 100  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 200  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

GOAL_SIZE = 16
STEP_SIZE = 4
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 256

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # , 20, 32]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]  # , [160, 160]
GOAL_OUT_SIZE = sum(dis_num_filters)

dis_dropout_keep_prob = 1.0
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 800
positive_file = 'save/realtrain_cotra.txt'
negative_file = 'save/generator_sample.txt'
generated_num = 10000
model_path = './ckpts'