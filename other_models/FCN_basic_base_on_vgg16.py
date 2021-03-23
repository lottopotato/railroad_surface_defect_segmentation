import os, time
import numpy as np

from keras import backend as k
from keras.models import Model, load_model
from keras.layers import Conv2D, Conv2DTranspose, add, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics, losses, optimizers
from keras.applications.vgg16 import VGG16

from sklearn import metrics as sk_metrics

from load_data import load_dataset, expand_all
from utils import compute_moreful_scores

def mean_softmax_crossentropy(y_true, y_pred):
	return k.mean(losses.categorical_crossentropy(y_true, y_pred))

def preprocess(data, shuffle = False, rescaled = False, mean = True, std = False):
	train_img, train_labels, test_img, test_labels = expand_all(data)

	if std:
		mean = True
	if rescaled:
		process = ImageDataGenerator(rescale = 1./255)
	else:
		process = ImageDataGenerator(featurewise_center = mean, 
			featurewise_std_normalization=std)

	print(' image preprocessing to rescale : ', str(rescaled), ' mean subtraction : ', str(mean),
		' std : ', str(std))
	process.fit(train_img)
	train_images = process.flow(train_img, batch_size = train_img.shape[0],
		shuffle = shuffle)[0]
	test_images = process.flow(test_img, batch_size = test_img.shape[0],
		shuffle = shuffle)[0]
	return {'train_img' : train_images, 'train_label' : train_labels,
	'test_img' : test_images, 'test_label' : test_labels}

def FCN_type1(show_summary = True):
	vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (100, 160, 3,))

	fc1 = Conv2D(filters = 4096, kernel_size = (1,1), padding = 'same', activation = 'relu')(vgg16.output)
	dropout1 = Dropout(rate = 0.5)(fc1)
	fc2 = Conv2D(filters = 4096, kernel_size = (1,1),
		padding = 'same', activation = 'relu')(dropout1)
	dropout2 = Dropout(rate = 0.5)(fc2)
	fc3 = Conv2D(filters = 2, kernel_size = (1,1),
		padding = 'same', activation = 'linear')(dropout2)
	deconv1 = Conv2DTranspose(filters = 2, kernel_size = (4,4),
		strides = 2, padding = 'same', activation = 'linear')(fc3)
	pool4_f = Conv2D(filters = 2, kernel_size = (1,1),
		padding = 'same', activation = 'linear')(vgg16.get_layer('block4_pool').output)
	fuse1 = add([deconv1, pool4_f])
	deconv2 = Conv2DTranspose(filters = 2, kernel_size = (4,4),
		strides = 2, padding = 'same', activation = 'linear')(fuse1)
	pool3_f = Conv2D(filters = 2, kernel_size=(1,1),
		padding = 'same', activation = 'linear')(vgg16.get_layer('block3_pool').output)
	fuse2 = add([deconv2, pool3_f])
	deconv3 = Conv2DTranspose(filters = 2, kernel_size = (12,8),
		strides = 8, padding = 'valid', activation = 'softmax')(fuse2)

	fcn_basic = Model(inputs = vgg16.input, outputs = deconv3)
	if show_summary:
		fcn_basic.summary()
	return fcn_basic

def FCN_type2(show_summary = True):
	vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (100, 55, 3,))

	fc1 = Conv2D(filters = 4096, kernel_size = (1,1), padding = 'same', activation = 'relu')(vgg16.output)
	dropout1 = Dropout(rate = 0.5)(fc1)
	fc2 = Conv2D(filters = 4096, kernel_size = (1,1),
		padding = 'same', activation = 'relu')(dropout1)
	dropout2 = Dropout(rate = 0.5)(fc2)
	fc3 = Conv2D(filters = 2, kernel_size = (1,1),
		padding = 'same', activation = 'linear')(dropout2)
	deconv1 = Conv2DTranspose(filters = 2, kernel_size = (2,3),
		strides = 2, padding = 'valid', activation = 'linear')(fc3)
	pool4_f = Conv2D(filters = 2, kernel_size = (1,1),
		padding = 'same', activation = 'linear')(vgg16.get_layer('block4_pool').output)
	fuse1 = add([deconv1, pool4_f])
	deconv2 = Conv2DTranspose(filters = 2, kernel_size = (2,2),
		strides = 2, padding = 'valid', activation = 'linear')(fuse1)
	pool3_f = Conv2D(filters = 2, kernel_size=(1,1),
		padding = 'same', activation = 'linear')(vgg16.get_layer('block3_pool').output)
	fuse2 = add([deconv2, pool3_f])
	deconv3 = Conv2DTranspose(filters = 2, kernel_size = (12,15),
		strides = 8, padding = 'valid', activation = 'softmax')(fuse2)

	fcn_basic = Model(inputs = vgg16.input, outputs = deconv3)
	if show_summary:
		fcn_basic.summary()
	return fcn_basic

def FCN_compile(fcn_model):
	optimizer = optimizers.Adam(lr=1e-4)
	loss = mean_softmax_crossentropy
	fcn_model.compile(optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
	return fcn_model

def FCN_running(data_type = 'type1', epochs = 300):
	history_name = 'fcn_basic_' + data_type + '.txt'

	if data_type == 'type1':
		fcn_model = FCN_type1()
	elif data_type == 'type2':
		fcn_model = FCN_type2()
	else:
		raise ValueError('..')
	
	dataset = load_dataset()
	dataset = preprocess(dataset[data_type], std = True)

	fcn_model = FCN_compile(fcn_model)

	fitting_validation(fcn_model, dataset, history_name, epochs, 16)
	
def fitting_validation(model, dataset, history_name, epochs, batch_size):
	sttime = time.time()
	model.fit(x = dataset['train_img'], y = dataset['train_label'],
		validation_data = (dataset['test_img'], dataset['test_label']), 
		epochs = epochs, batch_size = batch_size)
	fttime = str(time.time()-sttime)
	print(' >> total fitting time : ', fttime)

	compute_moreful_scores(model, dataset, history_name)
	return fttime

if  __name__ == "__main__":
	FCN_running()
	k.clear_session()