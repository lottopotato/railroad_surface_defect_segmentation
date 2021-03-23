import os, time
import numpy as np

import keras.models as models
from keras import backend as k
from keras import metrics, losses, optimizers
from keras.layers import Conv2D, Conv2DTranspose, add, Dropout, ReLU, Input, \
BatchNormalization, MaxPooling2D, UpSampling2D, Input, Reshape, Permute, Softmax
from sklearn import metrics as sk_metrics

from load_data import type2_makeup, list_to_arr, labeling, shuffle_with_sameindex_img_label, \
load_dataset, expand_ch3, load_data_all
from utils import compute_moreful_scores

def mean_softmax_crossentropy(y_true, y_pred):
	return k.mean(losses.categorical_crossentropy(y_true, y_pred))
def preprocess(dataset):
	dataset['train_img'] = expand_ch3(dataset['train_img'])
	dataset['test_img'] = expand_ch3(dataset['test_img'])
	dataset['train_label'] = labeling(dataset['train_label'], axis = -1)
	dataset['test_label'] = labeling(dataset['test_label'], axis = -1)
	return dataset


class segnet:
	def __init__(self, data_type, name = 'type1', data_dims = 2, epochs = 300, batch_size = 16, learning_rate = 1e-4):
		self.first_filter = 64
		self.data_type = data_type
		self.data_dims = data_dims
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = learning_rate
		self.model_name = 'segnet_model_' + name + '.h5'
		self.history_name = name + '.txt'

	def segnet_encoder(self, inputs):
		B1_Conv1 = Conv2D(filters = self.first_filter, kernel_size = (3,3), padding = 'same')(inputs)
		B1_BN1 = BatchNormalization()(B1_Conv1)
		B1_Relu1 = ReLU()(B1_BN1)
		B1_Conv1 = Conv2D(filters = self.first_filter, kernel_size = (3,3), padding = 'same')(B1_Relu1)
		B1_BN2 = BatchNormalization()(B1_Conv1)
		B1_Relu2 = ReLU()(B1_BN2)
		B1_Pool2 = MaxPooling2D(pool_size = (2,2), padding = 'same')(B1_Relu2)

		n_filter = int(self.first_filter*2)
		B2_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B1_Pool2)
		B2_BN1 = BatchNormalization()(B2_Conv1)
		B2_Relu1 = ReLU()(B2_BN1)
		B2_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B2_Relu1)
		B2_BN2 = BatchNormalization()(B2_Conv1)
		B2_Relu2 = ReLU()(B2_BN2)
		B2_Pool2 = MaxPooling2D(pool_size = (2,2), padding = 'same')(B2_Relu2)

		n_filter = int(n_filter*2)
		B3_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B2_Pool2)
		B3_BN1 = BatchNormalization()(B3_Conv1)
		B3_Relu1 = ReLU()(B3_BN1)
		B3_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B3_Relu1)
		B3_BN2 = BatchNormalization()(B3_Conv1)
		B3_Relu2 = ReLU()(B3_BN2)
		B3_Pool2 = MaxPooling2D(pool_size = (2,2), padding = 'same')(B3_Relu2)

		n_filter = int(n_filter*2)
		B4_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B3_Pool2)
		B4_BN1 = BatchNormalization()(B4_Conv1)
		B4_Relu1 = ReLU()(B4_BN1)
		B4_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding = 'same')(B4_Relu1)
		B4_BN2 = BatchNormalization()(B4_Conv1)
		B4_Relu2 = ReLU()(B4_BN2)
		B4_Pool2 = MaxPooling2D(pool_size = (2,2), padding = 'same')(B4_Relu2)

		return B4_Pool2, n_filter

	def decoder_for_type1(self, incoder, n_filter):
		B5_Upsp1 = UpSampling2D(size=(2,2))(incoder)
		B5_Conv1 = Conv2D(filters = n_filter, kernel_size = (2, 1), padding ='valid')(B5_Upsp1)
		B5_BN1 = BatchNormalization()(B5_Conv1)
		B5_Relu1 = ReLU()(B5_BN1)
		B5_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B5_Relu1)
		B5_BN2 = BatchNormalization()(B5_Conv2)
		B5_Relu2 = ReLU()(B5_BN2)

		n_filter = int(n_filter/2)
		B6_Upsp1 = UpSampling2D(size=(2,2))(B5_Relu2)
		B6_Conv1 = Conv2D(filters = n_filter, kernel_size = (2, 1), padding ='valid')(B6_Upsp1)
		B6_BN1 = BatchNormalization()(B6_Conv1)
		B6_Relu1 = ReLU()(B6_BN1)
		B6_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B6_Relu1)
		B6_BN2 = BatchNormalization()(B6_Conv2)
		B6_Relu2 = ReLU()(B6_BN2)

		n_filter = int(n_filter/2)
		B7_Upsp1 = UpSampling2D(size=(2,2))(B6_Relu2)
		B7_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B7_Upsp1)
		B7_BN1 = BatchNormalization()(B7_Conv1)
		B7_Relu1 = ReLU()(B7_BN1)
		B7_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B7_Relu1)
		B7_BN2 = BatchNormalization()(B7_Conv2)
		B7_Relu2 = ReLU()(B7_BN2)

		n_filter = int(n_filter/2)
		B8_Upsp1 = UpSampling2D(size=(2,2))(B7_Relu2)
		B8_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B8_Upsp1)
		B8_BN1 = BatchNormalization()(B8_Conv1)
		B8_Relu1 = ReLU()(B8_BN1)
		B8_Conv2 = Conv2D(filters = self.data_dims, kernel_size = (3,3), padding ='same')(B8_Relu1)
		B8_BN2 = BatchNormalization()(B8_Conv2)
		B8_SMax2 = Softmax()(B8_BN2)

		return B8_SMax2

	def decoder_for_type2(self, incoder, n_filter):
		B5_Upsp1 = UpSampling2D(size=(2,2))(incoder)
		B5_Conv1 = Conv2D(filters = n_filter, kernel_size = (2,2), padding ='valid')(B5_Upsp1)
		B5_BN1 = BatchNormalization()(B5_Conv1)
		B5_Relu1 = ReLU()(B5_BN1)
		B5_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B5_Relu1)
		B5_BN2 = BatchNormalization()(B5_Conv2)
		B5_Relu2 = ReLU()(B5_BN2)

		n_filter = int(n_filter/2)
		B6_Upsp1 = UpSampling2D(size=(2,2))(B5_Relu2)
		B6_Conv1 = Conv2D(filters = n_filter, kernel_size = (2,1), padding ='valid')(B6_Upsp1)
		B6_BN1 = BatchNormalization()(B6_Conv1)
		B6_Relu1 = ReLU()(B6_BN1)
		B6_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B6_Relu1)
		B6_BN2 = BatchNormalization()(B6_Conv2)
		B6_Relu2 = ReLU()(B6_BN2)

		n_filter = int(n_filter/2)
		B7_Upsp1 = UpSampling2D(size=(2,2))(B6_Relu2)
		B7_Conv1 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B7_Upsp1)
		B7_BN1 = BatchNormalization()(B7_Conv1)
		B7_Relu1 = ReLU()(B7_BN1)
		B7_Conv2 = Conv2D(filters = n_filter, kernel_size = (3,3), padding ='same')(B7_Relu1)
		B7_BN2 = BatchNormalization()(B7_Conv2)
		B7_Relu2 = ReLU()(B7_BN2)

		n_filter = int(n_filter/2)
		B8_Upsp1 = UpSampling2D(size=(2,2))(B7_Relu2)
		B8_Conv1 = Conv2D(filters = n_filter, kernel_size = (1,2), padding ='valid')(B8_Upsp1)
		B8_BN1 = BatchNormalization()(B8_Conv1)
		B8_Relu1 = ReLU()(B8_BN1)
		B8_Conv2 = Conv2D(filters = self.data_dims, kernel_size = (3,3), padding ='same')(B8_Relu1)
		B8_BN2 = BatchNormalization()(B8_Conv2)
		B8_SMax2 = Softmax()(B8_BN2)

		return B8_SMax2

	def models(self, first_filter = 64, show_summary = True):
		if self.data_type == 'type1':
			v1, v2 = 100, 160
			inputs_tensor = Input(shape = (v1, v2, 3,))
			_encoder, last_filter = self.segnet_encoder(inputs_tensor)
			_decoder = self.decoder_for_type1(_encoder, last_filter)

		elif self.data_type == 'type2':
			v1, v2 = 100, 55
			inputs_tensor = Input(shape = (v1, v2, 3,))
			_encoder, last_filter = self.segnet_encoder(inputs_tensor)
			_decoder = self.decoder_for_type2(_encoder, last_filter)

		else:
			raise ValueError('..')

		self.segnet_basic = models.Model(inputs = inputs_tensor, outputs = _decoder)

		optimizer = optimizers.Adam(lr=0.001)
		loss = mean_softmax_crossentropy
		self.segnet_basic.compile(optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
		if show_summary:
			self.segnet_basic.summary()

def segnet_for_mydata(data_type = 'type1', epochs = 300):
	segnet_model = segnet(data_type, name = '_segnet_', data_dims = 2, epochs = epochs, batch_size = 16, learning_rate = 1e-4)
	Segnet = segnet_model.models()

	dataset = load_dataset()[data_type]
	dataset = preprocess(dataset)

	history_name = 'segnet_basic_' + data_type + '.txt'

	fitting_validation(segnet_model, dataset, history_name, moreful_score = True)

def fitting_validation(model, dataset, history_name, moreful_score = True):
	sttime = time.time()
	segnet_model = model.segnet_basic
	segnet_model.fit(x = dataset['train_img'], y = dataset['train_label'],
		validation_data = (dataset['test_img'], dataset['test_label']), 
		epochs = model.epochs, batch_size = model.batch_size)
	fttime = str(time.time()-sttime)
	print(' >> total fitting time : ', fttime)

	if moreful_score:
		compute_moreful_scores(segnet_model, dataset, history_name)

	return fttime


def segnet_sample(epochs = 300):
	img_w = 1250
	img_h = 55
	n_labels = 2

	kernel = 3
	pool_size = 2

	encoding_layers = [
	    Conv2D(filters = 64, kernel_size = (kernel, kernel), padding ='same', input_shape = (1250, 55, 1,)),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 64, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    MaxPooling2D(pool_size=(pool_size, pool_size), padding = 'same'),

	    Conv2D(filters = 128, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 128, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    MaxPooling2D(pool_size=(pool_size, pool_size), padding = 'same'),

	    Conv2D(filters = 256, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 256, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    MaxPooling2D(pool_size=(pool_size, pool_size), padding = 'same'),

	    Conv2D(filters = 512, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 512, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    MaxPooling2D(pool_size=(pool_size, pool_size), padding = 'same'),
	    
	   
	]

	decoding_layers = [
	    UpSampling2D(size=(pool_size,pool_size)),
	    Conv2D(filters = 512, kernel_size = (2, 2), padding ='valid'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 512, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),

	    UpSampling2D(size=(pool_size,pool_size)),
	    Conv2D(filters = 256, kernel_size = (2, 1), padding ='valid'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 256, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),

	    UpSampling2D(size=(pool_size,pool_size)),
	    Conv2D(filters = 128, kernel_size = (2, 1), padding ='valid'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 128, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    
	    UpSampling2D(size=(pool_size,pool_size)),
	    Conv2D(filters = 64, kernel_size = (1,2), padding ='valid'),
	    BatchNormalization(),
	    ReLU(),
	    Conv2D(filters = 64, kernel_size = (kernel, kernel), padding ='same'),
	    BatchNormalization(),
	    ReLU(),
	    
	    Conv2D(filters = n_labels, kernel_size = (1, 1), padding ='valid'),
	    BatchNormalization(),
	    Softmax()
	]


	segnet_basic = models.Sequential()

	#segnet_basic.add(Input(shape=(360, 480, 3,)))
	segnet_basic.encoding_layers = encoding_layers
	for l in segnet_basic.encoding_layers:
	    segnet_basic.add(l)
	segnet_basic.decoding_layers = decoding_layers
	for l in segnet_basic.decoding_layers:
	    segnet_basic.add(l)


	optimizer = optimizers.Adam(lr=0.001)
	loss = mean_softmax_crossentropy
	segnet_basic.compile(optimizer = optimizer, loss = loss, metrics = ['categorical_accuracy'])
	segnet_basic.summary()

	_, _, t2_img, t2_lbl = load_data_all()
	img, label = np.asarray(t2_img), np.asarray(t2_lbl)
	
	img, label = type2_makeup(img, label, v1 = 1250, v2 = 55)
	img = np.expand_dims(list_to_arr(img), axis = -1)
	label = labeling(list_to_arr(label), axis = -1)
	print(img.shape, label.shape)

	img, label, _ = shuffle_with_sameindex_img_label(img, label)

	train_img, test_img = img[:-9], img[-9:]
	train_label, test_label = label[:-9], label[-9:]

	segnet_basic.fit(x = train_img, y = train_label,
		validation_data = (test_img, test_label), 
		epochs = epochs, batch_size = 4)

	prediction = (np.asarray(segnet_basic.predict(test_img)))[:,:,:,1].round().flatten()
	target = test_label[:,:,:,1].flatten()
	print(segnet_basic.evaluate(x = test_img, y = test_label))
	_f1_score = sk_metrics.f1_score(target, prediction)
	_recall = sk_metrics.recall_score(target, prediction)
	_precision = sk_metrics.precision_score(target, prediction)
	positive = np.where(np.logical_not((np.vstack((target, prediction)) == 0).all(axis=0))) # except 0
	_mean_iou = sk_metrics.jaccard_similarity_score(target[positive], prediction[positive])
	print(' - f1 score : %f | recall : %f | precision : %f | mean-IoU : %f'%(_f1_score, _recall, _precision, _mean_iou))

if  __name__ == "__main__":
	#segnet_for_mydata('type1')
	segnet_sample(epochs = 300)
	k.clear_session()