from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.models import load_model
import json
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy.stats import chi
from absl import flags
from absl import app

from utils import MammoData, CardioData, ThyroidData, LymphoData


FLAGS = flags.FLAGS

flags.DEFINE_bool('plot', False, 'Plot')
flags.DEFINE_bool('train', False, 'Train AAE')
flags.DEFINE_bool('anomaly_expt', False, 'Run anomaly detection experiment')
flags.DEFINE_enum('dataset', 'mammo', ['mammo', 'cardio', 'thyroid', 'lympho'], 'Dataset to use, from mammo, cardio, thyroid or lympho')
flags.DEFINE_integer('std', 10, 'std')
flags.DEFINE_integer('iter', 5, 'No. of iForest runs per sample set')
flags.DEFINE_integer('samples', 5, 'No. of sample sets generated')


def cal_auc(x, y):
	return np.trapz(y, x)

def get_dist(values):
	center = np.mean(values, axis=0)
	std = np.std(values, axis=0)
	chi_std = chi.std(2, 0, np.linalg.norm(std))
	dist = np.linalg.norm(values - center, axis=1)
	for i, el in enumerate(dist):
		if el > 3. * chi_std:
			dist[i] = 0.
	sorted_dist = np.copy(dist)
	sorted_dist.sort()
	threshold = sorted_dist[int(len(dist) * 0.9)]
	for i, el in enumerate(dist):
		if el < threshold:
			dist[i] = 0
	dist /= np.sum(dist)
	return dist

def roc_val(classifier, x_test, y_test):

	predictions = classifier.predict(x_test)

	predicted_anomalies = (predictions == -1).astype(np.int32)

	tp = np.sum(predicted_anomalies[np.where(y_test == predicted_anomalies)] == 1)
	tn = np.sum(predicted_anomalies[np.where(y_test == predicted_anomalies)] == 0)
	fp = np.sum(predicted_anomalies) - tp
	fn = np.sum(predicted_anomalies == 0) - tn

	if tp == 0:
		recall = tp_rate = 0.
		precision = 1.
	else:
		recall = tp_rate = tp / (tp + fn)
		precision = tp / (tp + fp)
	if recall + precision == 0:
		f1 = 0.
	else:
		f1 = (2 * recall * precision) / (recall + precision)
	fp_rate = fp / (fp + tn)

	return {'TPR': tp_rate, 'FPR': fp_rate, 'F1': f1}

def doping_edge(n_run):
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	latent = encoder.predict(x)
	
	dist = get_dist(latent)

	samples = []
	for i in np.arange(synth_size):
		choice = np.random.choice(np.arange(len(dist)), p=dist)
		samples.append(latent[choice])
	samples = decoder.predict(latent.reshape(-1, 2)).tolist()
	with open(folder + 'decoded_samples_{}.json'.format(FLAGS.std, n_run), 'w') as file:
		json.dump(samples, file)
	return samples

def generate(n_run):
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	latent = encoder.predict(x)
	center = np.mean(latent, axis=0)
	latent = np.random.randn(synth_size, 2)
	for i, vector in enumerate(latent):
		latent[i] = 10. * vector / np.linalg.norm(vector)
	latent += center
	samples = decoder.predict(latent.reshape(-1, 2))
	with open(folder + "synthetic_samples_{}.json".format(FLAGS.std, n_run), 'w') as file:
		json.dump(samples.tolist(), file)
	return samples

def smote(n_run):
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	
	samples = []
	for i in np.arange(synth_size):
		choice = np.random.choice(np.arange(len(x)))
		a = x[choice]
		x_copy = np.concatenate((x[:choice], x[choice + 1:]))
		x_copy -= a
		x_copy = np.linalg.norm(x_copy, axis=1)
		b = np.argmin(x_copy)
		if b >= choice:
			b += 1
		b = x[b]
		scale = np.random.rand()
		c = scale * (a-b) + b
		samples.append(list(c))
	with open(folder + 'smote_reg_data_samples_{}.json'.format(FLAGS.std, n_run), 'w') as file:
		json.dump(samples, file)
	return samples

def expt(n_run):
	(x_train, y_train), (x_test, y_test) = dataset.load_data()

	x_synth = {
		'doping': doping_edge(n_run),
		'smote': smote(n_run),
	}

	x = {
		'original': x_train,
	}
	for synth_type in x_synth:
		x[synth_type] = np.concatenate((x_train, x_synth[synth_type]))

	stat_types = ['TPR', 'FPR', 'F1']

	stats = {}
	for method in x:
		stats[method] = dict(zip(stat_types, [[] for stat in stat_types]))

	con_vals = np.arange(0.01, 0.3, 0.02)
	con_vals = np.concatenate(([0.001, 0.003, 0.005, 0.007], con_vals))
	for i, con_val in enumerate(con_vals):

		print('Run #{}/{}'.format(i + 1, len(con_vals)))

		run_stats = {}
		for method in x:
			run_stats[method] = dict(zip(stat_types, [[] for stat in stat_types]))

		for j in np.arange(FLAGS.iter):

			classifiers = {}
			for method in x:
				classifiers[method] = IsolationForest(contamination=con_val)
				classifiers[method].fit(x[method])

				results = roc_val(classifiers[method], x_test, y_test)
				for stat in results:
					run_stats[method][stat].append(results[stat])

		for method in stats:
			for stat in stat_types:
				stats[method][stat].append(np.mean(run_stats[method][stat]))

	return stats

def train():

	methods = ['original', 'doping', 'smote']
	stat_types = ['TPR', 'FPR', 'F1']
	all_stats = {}
	for method in methods:
		all_stats[method] = dict(zip(stat_types, [[] for stat in stat_types]))

	for i in np.arange(FLAGS.samples):
		expt_stats = expt(i)
		for method in methods:
			for stat in stat_types:
				all_stats[method][stat].append(expt_stats[method][stat])

	for method in methods:
		for stat in stat_types:
			all_stats[method][stat] = np.mean(all_stats[method][stat], axis=0).tolist()

	with open(folder + 'stats.json'.format(FLAGS.std), 'w') as file:
		json.dump(all_stats, file)

def plot(all_stats, methods=None):
	
	f1_list = []
	auc_list = []
	g_list = []
	if methods == None:
		methods = all_stats.keys()
	for method in methods:
		# print('\n' + method)
		f1 = np.max(all_stats[method]['F1'])
		auc = cal_auc(np.concatenate(([0.0], all_stats[method]['FPR'], [1.0])), np.concatenate(([0.0], all_stats[method]['TPR'], [1.0])))
		# print('F1[{}]\t{}'.format(np.argmax(all_stats[method]['F1']), np.max(all_stats[method]['F1'])))
		# print('AUC\t{}'.format(cal_auc(np.concatenate(([0.0], all_stats[method]['FPR'], [1.0])), np.concatenate(([0.0], all_stats[method]['TPR'], [1.0])))))
		f1_list.append([f1, method])
		auc_list.append([auc, method])

		r = all_stats[method]['TPR'][np.argmax(all_stats[method]['F1'])]
		p = f1 * r / (2 * r - f1)
		g = (r * p) ** 0.5
		# print(2 * p * r / (p + r))
		# print(p, r, f1)
		g_list.append([g, method])

	f1_list.sort(reverse=True)
	auc_list.sort(reverse=True)
	g_list.sort(reverse=True)
	print('\nF1:')
	for [f1, method] in f1_list:
		print('{}: {}'.format(method, f1))
	print('\nAUC:')
	for [auc, method] in auc_list:
		print('{}: {}'.format(method, auc))
	print('\nG:')
	for [g, method] in g_list:
		print('{}: {}'.format(method, g))

def main(unused_argv):

	global desc, folder, synth_size, encoder, decoder, dataset
	desc = 'aae'
	folder = './expt_std{}_temp2/'.format(FLAGS.std)
	folder = './'
	tf.gfile.MakeDirs(folder)
	encoder = load_model('{}_encoder_{}_test.h5'.format(desc, FLAGS.std))
	decoder = load_model('{}_decoder_{}_test.h5'.format(desc, FLAGS.std))
	if FLAGS.dataset == 'mammo':
		dataset = MammoData()
		synth_size = 1100
	elif FLAGS.dataset == 'cardio':
		dataset = CardioData()
		synth_size = 80
	elif FLAGS.dataset == 'thyroid':
		dataset = ThyroidData()
		synth_size = 200
	elif FLAGS.dataset == 'lympho':
		dataset = LymphoData()
		synth_size = 7

	if FLAGS.train:
		train()
	elif FLAGS.plot:
		methods = ['original', 'doping', 'smote']
		stat_types = ['TPR', 'FPR', 'F1']
		
		with open(folder + 'stats.json'.format(FLAGS.std), 'r') as file:
			all_stats = json.load(file)
		plot(all_stats, methods)


if __name__ == '__main__':
	app.run(main)
