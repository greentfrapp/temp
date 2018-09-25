try:
	raw_input
except:
	raw_input = input

from modules.toy_generator import GeneratorDatasetA
from modules.toy_generator import GeneratorDatasetB
from modules.toy_generator import GeneratorDatasetC
from modules.aae_labeled import LabeledAdversarialAutoencoder
from modules.anomaly_detectors import AnomalyDetectorExperiment

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime
import tensorflow as tf
import json
from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, 'Train AAE')
flags.DEFINE_bool('viz_latent', False, 'Visualize latent space')
flags.DEFINE_bool('viz_data', False, 'Visualize data space')
flags.DEFINE_bool('anomaly_expt', False, 'Run anomaly detection experiment')
flags.DEFINE_bool('plot_auc', False, 'Plot AUC')

flags.DEFINE_enum('dataset', 'A', ['A', 'B', 'C'], 'Toy dataset to use, from A, B or C')
flags.DEFINE_string('modelpath', None, 'Path to load model')


def train(dataset_type='A'):
	if dataset_type == 'A':
		generator = GeneratorDatasetA()
		data_dim = 2
	elif dataset_type == 'B':
		generator = GeneratorDatasetB()
		data_dim = 3
	elif dataset_type == 'C':
		generator = GeneratorDatasetC()
		data_dim = 2
	else:
		raise TypeError('Unknown dataset type')
		quit()
	dataset = generator.generate()
	aae = LabeledAdversarialAutoencoder(data_dim=data_dim, z_dim=2)
	aae.train(data=dataset, n_epochs=2000)

def visualize_data(dataset_type='A'):
	if dataset_type == 'A':
		generator = GeneratorDatasetA()
		data_dim = 2
	elif dataset_type == 'B':
		generator = GeneratorDatasetB()
		data_dim = 3
	elif dataset_type == 'C':
		generator = GeneratorDatasetC()
		data_dim = 2
	else:
		raise TypeError('Unknown dataset type')
		quit()
	dataset = generator.generate()
	anomalous_datasets = []
	nominal_datasets = []
	for idx, value in enumerate(dataset.values):
		if dataset.anomaly_labels[idx] == 1:
			nominal_datasets.append(value)
		if dataset.anomaly_labels[idx] == -1:
			anomalous_datasets.append(value)

	fig = plt.figure()
	if len(nominal_datasets[0]) == 2:
		ax = fig.add_subplot(1, 1, 1)
		ax.scatter(np.array(nominal_datasets)[:, 0], np.array(nominal_datasets)[:, 1], color='#e74c3c', label='Nominal', s=3)
		ax.scatter(np.array(anomalous_datasets)[:, 0], np.array(anomalous_datasets)[:, 1], color='#2c3e50', label='Anomalous', s=3)
	elif len(nominal_datasets[0]) == 3:
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.scatter(np.array(nominal_datasets)[:, 0], np.array(nominal_datasets)[:, 1], np.array(nominal_datasets)[:, 2], color='#e74c3c', label='Nominal', s=3)
		ax.scatter(np.array(anomalous_datasets)[:, 0], np.array(anomalous_datasets)[:, 1], np.array(anomalous_datasets)[:, 2], color='#2c3e50', label='Anomalous', s=3)
	ax.set_aspect('equal')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
	ax.set_title('Dataset {}'.format(dataset_type), fontsize=14)
	plt.show(block=False)
	raw_input("Enter to quit...")

def visualize_latent(modelpath, dataset_type='A'):
	if dataset_type == 'A':
		generator = GeneratorDatasetA()
		data_dim = 2
	elif dataset_type == 'B':
		generator = GeneratorDatasetB()
		data_dim = 3
	elif dataset_type == 'C':
		generator = GeneratorDatasetC()
		data_dim = 2
	else:
		raise TypeError('Unknown dataset type')
		quit()
	dataset = generator.generate()
	aae = LabeledAdversarialAutoencoder(data_dim=data_dim, z_dim=2)
	aae.load(modelpath)
	# aae.load('Results/AdversarialAutoencoder/2018-09-25 20-49-30.095657_2_100_2000_adversarial_autoencoder/saved_models/')
	nominal = {'x': [], 'y': []}
	anomalous = {'x': [], 'y': []}
	for idx, value in enumerate(dataset.values):
		latent_vector = aae.encode(value)
		if dataset.anomaly_labels[idx] == -1:
			anomalous['x'].append(latent_vector[0])
			anomalous['y'].append(latent_vector[1])
		elif dataset.anomaly_labels[idx] == 1:
			nominal['x'].append(latent_vector[0])
			nominal['y'].append(latent_vector[1])
	fig, ax = plt.subplots()
	ax.scatter(nominal['x'], nominal['y'], color='#e74c3c', label='Nominal')
	ax.scatter(anomalous['x'], anomalous['y'], color='#2c3e50', label='Anomalous')
	ax.text(.5,1.07,'Latent Space of Dataset {}'.format(dataset_type), fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	ax.legend(loc=3, frameon=True, fontsize=14)
	ax.axis('equal')
	ax.set_xlim(-150, 150)
	ax.set_ylim(-150, 150)
	plt.show(block=False)
	raw_input("Enter to quit...")

def train_anomaly_detectors_distribution(modelpath, data_dim, z_dim=2, contamination=0.05, random_seed=1, aae_dists=np.concatenate(([1.], np.arange(5., 101., 5.))), aae_sizes=[100], runs=10):

	start_time = datetime.datetime.now()
	prev_end_time = start_time

	expt = AnomalyDetectorExperiment()
	aae = LabeledAdversarialAutoencoder(data_dim=data_dim, z_dim=z_dim)
	aae.load(modelpath)
	expt.aae = aae
	total_runs_remaining = len(aae_dists) * len(aae_sizes) * runs

	results_folder = modelpath + '_results/+100'
	tf.gfile.MakeDirs(results_folder)

	for size in aae_sizes:
		for dist in aae_dists:
			for i in range(runs):
				expt.run(random_seed=random_seed, artificial_sample_size=size, artificial_sampling='distribution', artificial_sample_distribution=dist)
				# expt.run(random_seed=random_seed, artificial_sample_size=size, artificial_sampling='none', artificial_sample_distribution=dist)
				end_time = datetime.datetime.now()
				delta = end_time - prev_end_time
				total_delta = end_time - start_time
				prev_end_time = end_time
				total_runs_remaining -= 1
				print("Total Time Taken: {} seconds".format(total_delta.seconds))
				print("Estimated Time Remaining: {} seconds".format((total_runs_remaining * delta).seconds))	
			expt.save_results(savepath=os.path.join(results_folder, 'artificial_results_{}_s{}.json'.format(dist, size)))
			# expt.save_results(savepath=os.path.join(results_folder, 'original.json'.format(dist, size)))
			expt.results = []

def plot_auc(resultspath):
	detector = "Isolation Forest"
	
	"""
	with open('Toy10_Labeled_results/original.json', 'r') as file:
		original_results = json.load(file)
	if detector == "Robust Deep Autoencoder":
		original_results[detector]["Recall"].reverse()
		original_results[detector]["FP_rate"].reverse()
	y = np.concatenate([[0,0],[0,0], original_results[detector]["Recall"], [1,1]])
	x = np.concatenate([[0,0],[0,0], original_results[detector]["FP_rate"], [1,1]])
	original_auc = np.trapz(y, x)
	#"""
	
	auc_dists = []

	for filename in os.listdir(resultspath)[:]:
		if filename != '.DS_Store' and 'json' in filename:
			with open(os.path.join(resultspath, filename), 'r') as file:
				results = json.load(file)
			dist = float(filename.split('_')[2])
			y = np.concatenate([[0,0], results[detector]["Recall"], [1,1]])
			x = np.concatenate([[0,0], results[detector]["FP_rate"], [1,1]])
			auc = np.trapz(y, x)
			auc_dists.append([dist, auc])

	auc_dists.sort()
	auc = []
	dists = []
	for area in auc_dists:
		dists.append(area[0])
		auc.append(area[1])

	auc = auc[:]
	dists = dists[:]
	fig, ax = plt.subplots()
	ax.plot(dists, auc, color='#27ae60', label='Augmented Data', linewidth=3.)
	# ax.plot(dists, np.ones_like(auc) * original_auc, linestyle='--', color='#2c3e50', label='Original Data', linewidth=3.)
	ax.set_xlim(0, dists[-1])
	ax.text(.5,1.07,'Area-Under-Curve (AUC) against Magnitude', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	ax.text(.5,1.02,'Isolation Forest', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	ax.legend(loc=0, frameon=True, fontsize=14)
	ax.set_xlabel('Magnitude')
	ax.set_ylabel('AUC')
	
	plt.show(block=False)
	raw_input("Enter to quit...")


def main(unused_args):
	if FLAGS.train:
		train(dataset_type=FLAGS.dataset)
	elif FLAGS.viz_latent:
		visualize_latent(modelpath=FLAGS.modelpath, dataset_type=FLAGS.dataset)
	elif FLAGS.viz_data:
		visualize_data(dataset_type=FLAGS.dataset)
	elif FLAGS.anomaly_expt:
		train_anomaly_detectors_distribution(modelpath=FLAGS.modelpath, data_dim=2)
	elif FLAGS.plot_auc:
		plot_auc(resultspath=FLAGS.modelpath + '_results/+100')


if __name__ == "__main__":
	app.run(main)
