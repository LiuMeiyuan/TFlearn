#files
data_dir = 'data/resize/ISIC-2017_Training_Data'
gt_dir = 'data/resize/ISIC-2017_Training_Part1_GroundTruth'
model_save_path = 'output/'
#train_record_path = 'data/train.tfrecord'
#val_record_path = 'data/val.tfrecord'
#im_shape = (1856, 2720, 3)
#im_shape = (928, 1344, 3)
im_shape = (448, 640, 3)

#train
val_fraction = 0.2
epoch = 100
batch_size = 16 
normalize = True
starter_learning_rate = 0.01
decay_steps = 500
decay_rate = 0.96
train_steps = 100
model_save_step = 1000

#model
depth = 4
filters_first = 64

#tensorboard
visual_dir = '.'
