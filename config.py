#files
data_dir = 'data/ISIC-2017_Training_Data'
gt_dir = 'data/ISIC-2017_Training_Part1_GroundTruth'
model_save_path = 'output/'
train_record_path = 'data/train.tfrecord'
val_record_path = 'data/val.tfrecord'
#im_shape = (1856, 2720, 3)
im_shape = (928, 1344, 3)

#train
val_fraction = 0.2
epoch = 50
batch_size = 32 
normalize = True
learning_rate = 0.01

#model
depth = 4
filters_first = 64

#tensorboard
visual_dir = '.'
