"""
This python code is old version for paper,
 H. Kim, S. Lee and S. Han, "Railroad Surface Defect Segmentation Using a Modified Fully Convolutional Network," KSII Transactions on Internet and Information Systems
 , vol. 14, no. 12, pp. 4763-4775, 2020. DOI: 10.3837/tiis.2020.12.008.

"""


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # cpu only
## ----------
from keras.models import Model, load_model
from keras.layers import Conv2D, Conv2DTranspose, add, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend, metrics, losses, optimizers
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from sklearn import metrics as sk_metrics
from load_data import expand_ch3, expand_all, img_save
from utils import compute_moreful_scores

class custom_metrics(Callback):
	def __init__(self, history_name = 'train_history.txt', calculate_score = False):
		self.history_name = history_name
		self.calculate_score = calculate_score
	def on_train_begin(self, logs = {}):
		self.f1_score = []
		self.recall = []
		self.precision = []
		self.mean_iou = []
		self.loss = []
		self.valid_loss = []
		self.acc = []
		self.valid_acc = []
		self.iteration = 0

	def on_train_end(self, logs = {}):
		self.history = {'loss' : self.loss, 'valid_loss' : self.valid_loss,
			'acc' : self.acc, 'valid_acc' : self.valid_acc,
			'f1_score': self.f1_score, 'recall' : self.recall, 'precision' : self.precision, 
			'mean_IoU': self.mean_iou, 'iteration' : self.iteration}
		with open(self.history_name, 'w') as f:
			f.write(str(self.history))
		print(' >> total iteration : ', self.iteration)
		return
	def on_epoch_begin(self, epoch, logs = {}):
		return
	def on_epoch_end(self, epoch, logs = {}):
		if self.calculate_score:
			prediction = (np.asarray(self.model.predict(self.validation_data[0])))[:,:,:,1].round().flatten()
			target = self.validation_data[1][:,:,:,1].flatten()
			_f1_score = sk_metrics.f1_score(target, prediction)
			_recall = sk_metrics.recall_score(target, prediction)
			_precision = sk_metrics.precision_score(target, prediction)
			positive = np.where(np.logical_not((np.vstack((target, prediction)) == 0).all(axis=0))) # except 0
			_mean_iou = sk_metrics.jaccard_similarity_score(target[positive], prediction[positive])
			self.f1_score.append(_f1_score)
			self.recall.append(_recall)
			self.precision.append(_precision)
			self.mean_iou.append(_mean_iou)
			self.loss.append(logs.get('loss'))
			self.valid_loss.append(logs.get('val_loss'))
			self.acc.append(logs.get('categorical_accuracy'))
			self.valid_acc.append(logs.get('val_categorical_accuracy'))
			
			print(' - f1 score : %f | recall : %f | precision : %f | mean-IoU : %f'%(_f1_score, _recall, _precision, _mean_iou))
		
		return
	def on_batch_begin(self, batch, logs = {}):
		self.iteration += 1
		return
	def on_batch_end(self, batch, logs = {}):
		return

def preprocess(data, shuffle = False, rescaled = False, mean = True, std = False):
	train_img, train_labels, test_img, test_labels = expand_all(data)
	#print(np.max(train_labels), np.max(test_labels))

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

def softmax_ce_with_logit(y_true, y_pred):
	return backend.categorical_crossentropy(y_true, y_pred, from_logits = True)

def mean_softmax_crossentropy(y_true, y_pred):
	return backend.mean(losses.categorical_crossentropy(y_true, y_pred))

def mean_softmax_crossentropy_with_logit(y_true, y_pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels =  y_true, logits = y_pred))


class fcn_model:
	def __init__(self, data_type, name = 'type1', data_dims = 2, epochs = 300, batch_size = 16, learning_rate = 1e-4):
		self.dense_filter = 1024
		self.data_type = data_type
		self.data_dims = data_dims
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = learning_rate
		self.model_name = 'fcn_model_' + name + '.h5'
		self.history_name = name + '.txt'

	def vgg_model(self, usepreparams=True):
		from keras.applications.vgg19 import VGG19

		if usepreparams:
			weight = 'imagenet'
		else:
			weight = 'None'
		vgg_model = VGG19(include_top = False, weights = weight)
		return vgg_model

	def model_1(self, pre_model = 'vgg', show_summary = True):
		if pre_model == 'vgg':
			back_model = self.vgg_model()
	
		fc1 = Conv2D(filters = self.dense_filter, kernel_size = (1,1),
			padding = 'same', activation = 'relu')(back_model.get_layer('block4_pool').output)
		dropout1 = Dropout(rate = 0.5)(fc1)
		fc2 = Conv2D(filters = self.dense_filter, kernel_size = (1,1),
			padding = 'same', activation = 'relu')(dropout1)
		dropout2 = Dropout(rate = 0.5)(fc2)
		fc3 = Conv2D(filters = self.data_dims, kernel_size = (1,1),
			padding = 'same', activation = 'linear')(dropout2)

		deconv1 = Conv2DTranspose(filters = self.data_dims, kernel_size = (4,4),
			strides = 2, padding = 'same', activation = 'linear')(fc3)
		pool4_f = Conv2D(filters = self.data_dims, kernel_size = (1,1),
			padding = 'same', activation = 'linear')(back_model.get_layer('block3_pool').output)
		fuse1 = add([deconv1, pool4_f])
		deconv2 = Conv2DTranspose(filters = self.data_dims, kernel_size = (3,2),
			strides = 2, padding = 'valid', activation = 'linear')(fuse1)
		pool3_f = Conv2D(filters = self.data_dims, kernel_size=(1,1),
			padding = 'same', activation = 'linear')(back_model.get_layer('block2_pool').output)
		fuse2 = add([deconv2, pool3_f])
		deconv3 = Conv2DTranspose(filters = self.data_dims, kernel_size = (4,4),
			strides = 2, padding = 'same')(fuse2)
		## 12,20 -> 25,40
		last_deconv = Conv2DTranspose(filters = self.data_dims, kernel_size = (4,4),
			strides = 2, padding = 'same', activation = 'softmax')(deconv3)

		self._model = Model(inputs = back_model.input, outputs = last_deconv)
		if show_summary:
			self.summary()

	def model_2(self, pre_model = 'vgg', show_summary = True):
		if pre_model == 'vgg':
			back_model = self.vgg_model()
	
		fc1 = Conv2D(filters = self.dense_filter, kernel_size = (1,1),
			padding = 'same', activation = 'relu')(back_model.get_layer('block4_pool').output)
		dropout1 = Dropout(rate = 0.5)(fc1)
		fc2 = Conv2D(filters = self.dense_filter, kernel_size = (1,1),
			padding = 'same', activation = 'relu')(dropout1)
		dropout2 = Dropout(rate = 0.5)(fc2)
		fc3 = Conv2D(filters = self.data_dims, kernel_size = (1,1),
			padding = 'same', activation = 'linear')(dropout2)

		deconv1 = Conv2DTranspose(filters = self.data_dims, kernel_size = (4,4),
			strides = 2, padding = 'same', activation = 'linear')(fc3)
		pool3_f = Conv2D(filters = self.data_dims, kernel_size = (1,1),
			padding = 'same', activation = 'linear')(back_model.get_layer('block3_pool').output)
		fuse1 = add([deconv1, pool3_f])
		# 12, 6 -> 25, 13
		deconv2 = Conv2DTranspose(filters = self.data_dims, kernel_size = (3,3),
			strides = 2, padding = 'valid', activation = 'linear')(fuse1)
		pool2_f = Conv2D(filters = self.data_dims, kernel_size=(1,1),
			padding = 'same', activation = 'linear')(back_model.get_layer('block2_pool').output)
		fuse2 = add([deconv2, pool2_f])
		# 25, 13 -> 50, 26
		deconv3 = Conv2DTranspose(filters = self.data_dims, kernel_size = (2,3),
			strides = 2, padding = 'valid')(fuse2)
		## 50,26 -> 100, 52
		last_deconv = Conv2DTranspose(filters = self.data_dims, kernel_size = (2,3),
			strides = 2, padding = 'valid', activation = 'softmax')(deconv3)

		self._model = Model(inputs = back_model.input, outputs = last_deconv)
		if show_summary:
			self.summary()


	def summary(self, line_length = 150):
		return self._model.summary(line_length = line_length)

	def compile_fit(self, datasets, test = False, model_save = True):
		self.custom_metrics = custom_metrics(self.history_name, calculate_score = test)
		optimizer = optimizers.Adam(lr=self.lr)
		loss = mean_softmax_crossentropy
		self._model.compile(optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
		self.history = self._model.fit(x = datasets['train_img'], y = datasets['train_label'],
			validation_data = (datasets['test_img'], datasets['test_label']), 
			epochs = self.epochs, batch_size = self.batch_size, callbacks = [self.custom_metrics])
		if model_save:
			self._model.save(self.model_name)

	def validation(self, dataset):
		compute_moreful_scores(self._model, dataset, self.history_name)

	def load_model_validation(self, dataset):
		loaded_model = load_model(self.model_name,
			custom_objects = {
			'mean_softmax_crossentropy' : mean_softmax_crossentropy,
			'optimizer' : optimizers.Adam(lr=self.lr)
			})

		compute_moreful_scores(loaded_model, dataset, self.history_name)

		
	def choice_sample(self, datasets, sample, full):
		valid_len = datasets['test_label'].shape[0]
		if not full:
			_index = np.random.choice(valid_len, sample, replace = False)
			_, v1, v2, v3 = datasets['test_img'].shape
			img = np.zeros((sample, v1, v2, v3))
			label = np.zeros((sample, v1, v2, self.data_dims))
			for i in range(sample):
				img[i], label[i]  = datasets['test_img'][_index[i]],  datasets['test_label'][_index[i]]
		else:
			img, label = datasets['test_img'], datasets['test_label']
			sample = valid_len
		return img, label, sample

	def result_images_save(self, predict, labels, length, root, pred_name = '_pred_', true_name = '_gt_', rescale = True):
		for i in range(length):
			prediction_name = self.data_type + pred_name + str(i)
			ground_truths_name = self.data_type + true_name + str(i)
			#img_save(array:np.ndarray, name, rescale = True, mode = None, root = 'custom')
			img_save(array = backend.eval(backend.argmax(predict[i], axis = 2)), name = prediction_name, root = root, rescale = rescale)
			img_save(array = backend.eval(backend.argmax(labels[i], axis = 2)), name = ground_truths_name, root = root, rescale = rescale)


	def visualize(self, datasets, sample = 10, full = False, rescale = True):
		img, label, length = self.choice_sample(datasets, sample, full)
		pred_img = self._model.predict(x = img)
		self.result_images_save(pred_img, label, length, 'FCN_PREDICT', '_pred_', '_gt_', rescale)

	def model_load_prediction(self, datasets, sample = 20, full = True, root = 'FCN_PREDICT', rescale = True):
		img, label, length = self.choice_sample(datasets, sample, full)
		loaded_model = load_model(self.model_name,
			custom_objects = {
			'mean_softmax_crossentropy' : mean_softmax_crossentropy,
			'optimizer' : optimizers.Adam(lr=self.lr)
			})
		pred_img = loaded_model.predict(x = img)
		self.result_images_save(pred_img, label, length, root, '_pred_', '_gt_', rescale)
		return length

	def predict(self, image, name = '_test'):
		save_name = self.data_type + name
		## test one
		image = np.expand_dims(image, axis=0)
		pred = backend.eval(backend.argmax(self._model.predict(x = image)[0], axis = 2))
		#pred = np.squeeze(validation, axis= 2)
		print(pred)
		img_save(pred, save_name)

	def predict_for_external_data(self, images, save_name = '_test', root = 'none'):
		try:
			loaded_model = load_model(self.model_name,
				custom_objects = {
				'mean_softmax_crossentropy' : mean_softmax_crossentropy,
				'optimizer' : optimizers.Adam(lr=self.lr)
				})
		except:
			raise ValueError(' The model name is not correct.')
		print(' # of iamges : ', len(images))
		pred = loaded_model.predict(x = images)
		if not root == 'none':
			print('save root : ', root)
		for i in range(len(images)):
			img_name = self.data_type + save_name + str(i)
			img_save(backend.eval(backend.argmax(pred[i], axis = 2)), img_name, root = root)
			print(' prediction saved %d/%d'%(i+1, len(images)), end = '\r')
		print('')	


	def show_history(self):
		from keras.utils import plot_model
		#plot_model(self._model, to_file = 'model.png')
		try:
			import matplotlib.pyplot as plt
			_, graph = plt.subplots()
			graph.plot(self.history.history['loss'], 'y', label = 'loss')
			graph.plot(self.history.history['categorical_accuracy'], 'y', label = 'accuracy')
			graph.set_xlabel('epoch')
			graph.set_legend()
			plt.show()
		except:
			print(' can not use pyplot ')
			#print( self.history.history)


		
