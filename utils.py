import numpy as np
from sklearn import metrics as sk_metrics

def compute_moreful_scores(model, dataset, history_name, check_nan = False):
	prediction = (np.asarray(model.predict(dataset['test_img'])))[:,:,:,1].round().flatten()
	target = dataset['test_label'][:,:,:,1].flatten()
	
	if check_nan:
		if np.isnan(np.sum(prediction)):
			nan_indices = np.argwhere(np.isnan(prediction))
			prediction[nan_indices] = 0
			print(' nan exist')
			
	evaluate = model.evaluate(x = dataset['test_img'], y = dataset['test_label'])
	print(np.mean(target), np.mean(prediction))
	_f1_score = sk_metrics.f1_score(target, prediction)
	_recall = sk_metrics.recall_score(target, prediction)
	_precision = sk_metrics.precision_score(target, prediction)
	positive = np.where(np.logical_not((np.vstack((target, prediction)) == 0).all(axis=0))) # except 0
	_mean_iou = sk_metrics.jaccard_similarity_score(target[positive], prediction[positive])
	print(' test loss : %f | test acc : %f'%(evaluate[0], evaluate[1]))
	print(' - f1 score : %f | recall : %f | precision : %f | mean-IoU : %f'%(_f1_score, _recall, _precision, _mean_iou))

	history = {'valid_loss' : evaluate[0], 'valid_acc' : evaluate[1],
		'f1_score': _f1_score, 'recall' : _recall, 'precision' : _precision, 
		'mean_IoU': _mean_iou}
	with open(history_name, 'w') as f:
		f.write(str(history))