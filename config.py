#files
data_dir = 'data/ISIC-2017_Training_Data'
gt_dir = 'data/ISIC-2017_Training_Part1_GroundTruth'
model_save_path = 'output/'

#train
validation_fraction = 0.2
epoch = 50
batch_size = 1
normalize = True
learning_rate = 0.01

#model
depth = 4
filters_first = 64

#tensorboard
visual_dir = '.'