from modules.generator import GeneratorToy1
from modules.generator import GeneratorToy2
from modules.generator import GeneratorToy3
from modules.generator import GeneratorToy4
from modules.generator import GeneratorToy5
from modules.generator import GeneratorToy6
from modules.generator import GeneratorToy7
from modules.generator import GeneratorToy8
from modules.aae_util import BasicAdversarialAutoencoder
from modules.aae_labeled import LabeledAdversarialAutoencoder
from modules.anomaly_detectors import AnomalyDetectorExperiment
import os
import numpy as np
import datetime

def train_anomaly_detectors_distribution(contamination, random_seed, data_dim, z_dim, modelpath, aae_dists, aae_sizes, runs):

	start_time = datetime.datetime.now()
	prev_end_time = start_time

	expt = AnomalyDetectorExperiment()
	aae = LabeledAdversarialAutoencoder(data_dim=data_dim, z_dim=z_dim)
	aae.load(os.path.join('./models', modelpath))
	expt.aae = aae
	total_runs_remaining = len(aae_dists) * len(aae_sizes) * runs

	results_folder = modelpath + '_results/+100'
	if not os.path.isdir(results_folder):
		if not os.path.isdir(modelpath + '_results'):
			os.mkdir(modelpath + '_results')
		os.mkdir(results_folder)

	for size in aae_sizes:
		for dist in aae_dists:
			for i in range(runs):
				#expt.run(random_seed=random_seed, artificial_sample_size=size, artificial_sampling='distribution', artificial_sample_distribution=dist)
				expt.run(random_seed=random_seed, artificial_sample_size=size, artificial_sampling='none', artificial_sample_distribution=dist)
				end_time = datetime.datetime.now()
				delta = end_time - prev_end_time
				total_delta = end_time - start_time
				prev_end_time = end_time
				total_runs_remaining -= 1
				print("Total Time Taken: {} seconds".format(total_delta.seconds))
				print("Estimated Time Remaining: {} seconds".format((total_runs_remaining * delta).seconds))	
			#expt.save_results(savepath=os.path.join(results_folder, 'artificial_results_{}_s{}.json'.format(dist, size)))
			expt.save_results(savepath=os.path.join(results_folder, 'original.json'.format(dist, size)))
			expt.results = []

if __name__ == "__main__":
	random_seed = 1
	contamination = 0.05
	z_dim = 2
	modelpath = 'Toy3_Labeled'
	data_dim = 2
	aae_dists = np.concatenate(([1.], np.arange(5., 101., 5.)))
	aae_dists = [0]
	aae_sizes = [100]
	runs = 10
	train_anomaly_detectors_distribution(data_dim=data_dim, contamination=contamination, random_seed=random_seed, z_dim=z_dim, modelpath=modelpath, aae_dists=aae_dists, aae_sizes=aae_sizes, runs=runs)
	
