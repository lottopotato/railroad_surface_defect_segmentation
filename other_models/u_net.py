import numpy as np
import os, time
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

from load_data import load_dataset, subsets_loaded_generated_images, type2_data_process
from utils import compute_moreful_scores
from fcn_model import preprocess, custom_metrics, mean_softmax_crossentropy

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + 1.) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + 1.)

def dice_coef_loss(y_true, y_pred):
	return 1.-dice_coef(y_true, y_pred)

def Unet_2d(img_rows, img_cols, start_neurons = 32, d_type = 'type1'):
	inputs = Input((img_rows, img_cols, 3,))
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
	conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)
	#pool1 = Dropout(0.25)(pool1)

	conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
	conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)
	#pool2 = Dropout(0.5)(pool2)

	conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
	conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
	pool3 = MaxPooling2D((2, 2))(conv3)
	#pool3 = Dropout(0.5)(pool3)

	conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
	conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
	pool4 = MaxPooling2D((2, 2))(conv4)
	#pool4 = Dropout(0.5)(pool4)

	# Middle
	convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
	convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
	
	deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(0.5)(uconv4)
	uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
	uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
	if d_type == 'type1':
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 2), strides=(2, 2), padding="valid")(uconv4)
	else:
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Dropout(0.5)(uconv3)
	uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
	uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
	if d_type == 'type1':
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
	else:
		deconv2 = Conv2DTranspose(start_neurons * 2, (2, 3), strides=(2, 2), padding ="valid")(uconv3)
	uconv2 = concatenate([deconv2, conv2])
	uconv2 = Dropout(0.5)(uconv2)
	uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
	uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
	if d_type == 'type1':
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
	else:
		deconv1 = Conv2DTranspose(start_neurons * 1, (2, 3), strides=(2, 2), padding="valid")(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	uconv1 = Dropout(0.5)(uconv1)
	uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
	
	output_layer = Conv2D(2, (1,1), padding="same", activation="softmax")(uconv1)

	model = Model(input=inputs, output=output_layer)
	loss = mean_softmax_crossentropy
	model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics = ['categorical_accuracy'])
	model.summary()

	return model

def compile_fit(model, datasets, history_name, model_name, epochs = 10, batch_size = 16, test = False, model_save = True):
	custom_callback = custom_metrics(history_name, calculate_score = test)
	sttime = time.time()

	history = model.fit(x = datasets['train_img'], y = datasets['train_label'],
		validation_data = (datasets['test_img'], datasets['test_label']), 
		epochs = epochs, batch_size = batch_size, callbacks = [custom_callback])
	compute_moreful_scores(model, datasets, history_name, check_nan = True)
	fttime = str(time.time()-sttime)
	print(' >> total fitting time : ', fttime)
	return fttime

def validate(d_type = 'type1'):
	railroad = load_dataset()
	data = preprocess(railroad[d_type], std = True)
	if d_type == 'type1':
		col = 160
		history_name = 'unet_t1e300_bs16_lr1e-4.txt'
	elif d_type == 'type2':
		col = 55
		history_name = 'unet_t2e300_bs16_lr1e-4.txt'
	print(data['train_img'].shape, data['train_label'].shape, data['test_img'].shape, data['test_label'].shape)
	unet = Unet_2d(img_rows = 100, img_cols = col)
	compile_fit(unet, data, history_name, 'Unet')

if  __name__ == "__main__":
	validate(d_type = 'type1')
	K.clear_session()
