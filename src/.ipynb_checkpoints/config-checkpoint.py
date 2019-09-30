#GAN model values
img_size = 128
batch_size = 128
epochs = 100

num_gen_mask = 100



# Paths
path_img = '../data/raw/ISIC2018_Task1-2_Training_Input'
path_mask = '../data/raw/ISIC2018_Task1_Training_GroundTruth'
path_models = '../models/'

path_trainingset = '../data/raw/train_segmentation.csv'
path_testset = '../data/raw/test_segmentation.csv'

path_gen = '../data/preprocessed'
