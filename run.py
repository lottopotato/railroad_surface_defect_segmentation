from load_data import load_dataset, subsets_loaded_generated_images, type2_data_process
from fcn_model import *
from keras import backend as k

import time
import sys

def validate():
	railroad = load_dataset()
	Type1 = fcn_model(data_type = 'type1', name = 't1e100_bs16_lr1e-4')
	Type2 = fcn_model(data_type = 'type1', name = 't2e300_bs16_lr1e-4')

	t1_data = preprocess(railroad['type1'], std = True)
	t2_data = preprocess(railroad['type2'], std = True)

	Type1.load_model_validation(t1_data['test_img'], t1_data['test_label'])
	Type2.load_model_validation(t2_data['test_img'], t2_data['test_label'])

def running(argv = 'null', _type = 'type1', epochs = 300):
	if not argv == 'null':
		data_type = argv[1]
	else:
		data_type = _type
	# 1. data prepare
	railroad = load_dataset()

	if data_type == 'type1':
		type1(railroad['type1'], epochs = epochs)
	elif data_type == 'type2':
		type2(railroad['type2'], epochs = epochs)
	else:
		raise ValueError(' type is \'type1\' or \'type2\' ')

def type1(data, epochs, weight_save_name = 't1e300_bs16_lr1e-4'):
	# 2. preprocess
	t1_data = preprocess(data, std = True)

	# 3. model
	FCN = fcn_model(data_type = 'type1', name = weight_save_name, epochs = epochs, batch_size = 16, learning_rate = 1e-4)
	
	FCN.model_1(pre_model = 'vgg')
	
	sttime = time.time()
	
	FCN.compile_fit(datasets = t1_data, test = False)
	fttime = str(time.time()-sttime)
	print(' >> total fitting time : ', fttime)
	with open('run_logs.txt', 'a') as f:
			f.write('\n' + str('days :'+ str(sttime) + ' name : ' + FCN.model_name + ' time : ' + str(fttime)))
	# 4. validation
	FCN.validation(t1_data)
	
	#FCN.load_model_validation(t1_data)

	# 4-option1. prediction
	#FCN.predict(t1_data['test_img'][6])
	# 4-option2. visualize
	#FCN.visualize(t1_data, sample = 50, full = False)
	#FCN.model_load_prediction(t1_data, sample = 20, full  =False)
	# 4-option3. history
	#FCN.show_history()

def type2(data, epochs, weight_save_name = 't2e300_bs16_lr1e-4'):
	# 2. preprocess
	t2_data = preprocess(data, std = True)

	# 3. model
	FCN = fcn_model(data_type = 'type2', name = weight_save_name, epochs = epochs, batch_size = 16, learning_rate = 1e-4)
	
	FCN.model_2(pre_model = 'vgg')
	sttime = time.time()
	FCN.compile_fit(datasets = t2_data, test = False)
	fttime = str(time.time()-sttime)
	print(' >> total fitting time : ', fttime)
	with open('run_logs.txt', 'a') as f:
			f.write('\n' + str('days :'+ str(sttime) + ' name : ' + FCN.model_name + ' time : ' + str(fttime)))

	# 4. validation
	FCN.validation(t2_data)
	
	#FCN.load_model_validation(t2_data['test_img'], t2_data['test_label'])

	# 4-option1. prediction
	#FCN.predict(t2_data['test_img'][1])
	# 4-option2. visualize
	#FCN.visualize(t2_data, sample = 20, full = True)
	# 4-option2. history
	#FCN.show_history()

def subset_run(argv = 'null', _type = 'type1', epochs = 300):
	if not argv == 'null':
		data_type = argv[1]
	else:
		data_type = _type
	# 1. data prepare
	railroad = subsets_loaded_generated_images(used_generated=True, data_type = data_type)
	if data_type == 'type2':
		railroad['type2'] = type2_data_process(railroad['type2'])

	if data_type == 'type1':
		type1(railroad['type1'], epochs = epochs, weight_save_name = 'exp5_gan_t1')
	elif data_type == 'type2':
		type2(railroad['type2'], epochs = epochs, weight_save_name = 'exp3_gan_t2')
	else:
		raise ValueError(' type is \'type1\' or \'type2\' ')


if  __name__ == "__main__":
	running(_type = 'type1', epochs = 3e+3)
	#running(sys.argv)
	#subset_run(sys.argv)
	#validate()
	k.clear_session()
