from __future__ import division

import tensorflow as tf 

import config as cfg 

def Downsampling_block(inputs, filters):
	x = conv_group(inputs, filters)
	pool_1 = tf.layers.max_pooling2d(x, 2, 2)
	return x, pool_1

def Upsampling_block(input_down, input_up, filters):
	conv_trans = tf.layers.conv2d_transpose(input_up, filters, 2, strides = 2)
	# wrong
	# if input_down.get_shape().as_list()[1] != None :
	# 	target_height = conv_trans.get_shape().as_list()[1]
	# 	target_width = conv_trans.get_shape().as_list()[2]
	# 	offset_height = int((input_down.get_shape().as_list()[1] - target_height) / 2)
	# 	offset_width = int((input_down.get_shape().as_list()[2] - target_width) / 2)
	# 	input_down = tf.image.crop_to_bounding_box(input_down, offset_height, offset_width, target_height, target_width)

	down_feature_shape = tf.shape(input_down)
	up_feature_shape = tf.shape(conv_trans)
	if down_feature_shape[1] % 2 != 0 :
		# height_pad = down_feature_shape[1].value -up_feature_shape[1].value
		# width_pad = down_feature_shape[2].value- up_feature_shape[2].value
		conv_trans = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 0))).apply(conv_trans)
	if down_feature_shape[2] % 2 != 0:
		conv_trans = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 1))).apply(conv_trans)

	x = tf.concat([input_down, conv_trans], axis = 3)
	x = conv_group(x, filters)
	return x

def conv_group(inputs, filters):
	x = tf.layers.conv2d(inputs, filters, 3,  padding = 'same')
	x = tf.nn.relu(x)
	x = tf.layers.conv2d(x, filters, 3,  padding = 'same')
	x = tf.nn.relu(x)
	return x

def u_net(x, y_gt):

	depth = cfg.depth
	filters = cfg.filters_first
	downsample_output = []

	for i in range(depth):
		down_feature_map, x = Downsampling_block(x, filters)
		downsample_output.append(down_feature_map)
		filters = filters * 2

	x = conv_group(x, filters)
	filters = int(filters / 2)

	for i in range(depth):
		x = Upsampling_block(downsample_output[depth - i - 1], x, filters)
		filters = int(filters / 2)

	logits = tf.layers.conv2d(x, 2, 1)

	predictions = tf.argmax(logits, axis = 3)

	loss = tf.losses.softmax_cross_entropy(y_gt, logits)
	return predictions, loss

		


