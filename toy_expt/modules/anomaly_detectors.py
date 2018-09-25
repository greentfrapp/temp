from modules.generator import GeneratorDatasetA
from modules.generator import GeneratorDatasetB
from modules.generator import GeneratorDatasetC
from modules.generator import Dataset
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class AnomalyDetectorBundle(object):

	def __init__(self, random_seed=1, outlier_fraction=0.05):
		self._detectors = dict()
		self._detectors['Isolation Forest'] = IsolationForest(contamination=outlier_fraction)
		self.random_seed = random_seed

	@property
	def random_seed(self):
		return self._random_seed

	@random_seed.setter
	def random_seed(self, value):
		assert isinstance(value, int)
		self._random_seed = value
		for detector_type, detector in self._detectors.items():
			detector.random_state = self.random_seed

	@property
	def outlier_fraction(self):
		return self._outlier_fraction

	@outlier_fraction.setter
	def outlier_fraction(self, value):
		assert isinstance(value, (int, float)) and value <= 1.
		self._outlier_fraction = value
		for detector_type, detector in self._detectors.items():
			detector.contamination = self.outlier_fraction

	def fit(self, dataset):
		for detector_type, detector in self._detectors.items():
			detector.fit(dataset)

	def predict(self, dataset):
		results = dict()
		for detector_type, detector in self._detectors.items():
			results[detector_type] = detector.predict(dataset)
		return results

class AnomalyDetectorExperiment(object):

	def __init__(self, dataset='A', size=1000, contamination=0.05, outlier_fractions=np.arange(0.01, 0.7, 0.04)):
		self.outlier_fractions = outlier_fractions
		if dataset == 'A':
			self._generator = GeneratorDatasetA()
		elif dataset == 'B':
			self._generator = GeneratorDatasetB()
		elif dataset == 'C':
			self._generator = GeneratorDatasetC()
		else:
			raise TypeError('Unknown dataset type')
			quit()
		self._detectors = AnomalyDetectorBundle()
		self.results = []
		self.size = size
		self.aae = None

	@property
	def size(self):
		return self._size

	@size.setter
	def size(self, value):
		assert isinstance(value, int)
		self._size = value

	@property
	def nominal_labels(self):
		return self._nominal_labels

	@nominal_labels.setter
	def nominal_labels(self, value):
		assert isinstance(value, (list, np.ndarray))
		self._nominal_labels = value

	@property
	def outlier_fractions(self):
		return self._outlier_fractions

	@outlier_fractions.setter
	def outlier_fractions(self,value):
		assert isinstance(value, (list, np.ndarray))
		self._outlier_fractions = value

	@property
	def results(self):
		return self._results

	@results.setter
	def results(self, value):
		self._results = value

	@property
	def aae(self):
		return self._aae

	@aae.setter
	def aae(self, value):
		self._aae = value
	
	def run(self, random_seed=1, artificial_sampling='none', samplespath=None, artificial_sample_size=50, artificial_sample_distribution=0.1):

		assert artificial_sampling in ['none', 'distribution']
		self.results.append(dict())
		for detector_type in self._detectors._detectors:
			self.results[-1][detector_type] = dict()

		print("Generating Datasets...")
		original_training_data = self._generator.generate(random_seed=random_seed)
		test_data = self._generator.generate(random_seed=random_seed+1)
		if artificial_sampling == 'none':
			training_data = original_training_data.values
			#lof_test_data = original_training_data.merge_with(test_data)
			#lof_test_data = Dataset()
		else:
			if artificial_sampling == 'distribution':
				if self.aae is None:
					print("Error: AAE not set")
					quit()
				artificial_training_data = self.aae.generate(mode='distribution', size=artificial_sample_size, distribution=artificial_sample_distribution, random_seed=random_seed)
			training_data = np.concatenate((original_training_data.values, artificial_training_data.values))
			#lof_test_data = original_training_data.merge_with(test_data).merge_with(artificial_training_data)
			#lof_test_data = Dataset()
		
		self._detectors.random_seed = random_seed

		# vary fractions
		for i, fraction in enumerate(self.outlier_fractions):
			print("{}/{} Outlier Fraction: {}".format(i + 1, len(self.outlier_fractions), fraction))
			self._detectors.outlier_fraction = fraction
			self._detectors.fit(training_data)
			results = self._detectors.predict(test_data.values)
			self.update_results(results, test_data.anomaly_labels)

		return self.results[-1]

	def update_results(self, detector_predictions, labels):

		for detector_type, predictions in detector_predictions.items():
			stats = dict()
			assert len(predictions) == len(labels)
			stats['TP'] = ((predictions == labels) * (predictions == -1)).sum()
			stats['TN'] = ((predictions == labels) * (predictions == 1)).sum()
			stats['FP'] = ((predictions != labels) * (predictions == -1)).sum()
			stats['FN'] = ((predictions != labels) * (predictions == 1)).sum()
			assert stats['TP'] + stats['TN'] + stats['FP'] + stats['FN'] == len(predictions)

			stats['Contamination'] = self._detectors._detectors[detector_type].contamination

			if (stats['TP'] + stats['FP']) == 0:
				stats['Precision'] = 1.
			else:
				stats['Precision'] = 1. * stats['TP'] / (stats['TP'] + stats['FP'])
			stats['Recall'] = 1. * stats['TP'] / (stats['TP'] + stats['FN'])
			if (stats['Precision'] + stats['Recall']) == 0:
				stats['F1'] = 0.
			else:
				stats['F1'] = 2. * (stats['Precision'] * stats['Recall']) / (stats['Precision'] + stats['Recall'])
			stats['FP_rate'] = 1. * stats['FP'] / (stats['FP'] + stats['TN'])
			stats['Accuracy'] = 100. * (stats['TP'] + stats['TN']) / (stats['TP'] + stats['TN'] + stats['FP'] + stats['FN'])

			if len(self.results[-1][detector_type]) == 0:
				for stat, stat_value in stats.items():
					self.results[-1][detector_type][stat] = [stat_value]
			else:
				for stat, stat_value in stats.items():
					self.results[-1][detector_type][stat].append(stat_value)

	def avg_results(self):
		avg_results = deepcopy(self.results[0])
		for idx, result in enumerate(self.results):
			for detector_type, detector_result in result.items():
				for stat, stat_value in detector_result.items():
					if idx == 0:
						avg_results[detector_type][stat] = [stat_value]
					else:
						avg_results[detector_type][stat].append(stat_value)
		for detector_type, detector_result in avg_results.items():
			for stat, stat_value in detector_result.items():
				avg_results[detector_type][stat] = np.mean(stat_value, axis=0).tolist()
		return avg_results

	def save_results(self, savepath='./results.json', run=None):
		print("Exporting results to {}".format(savepath))
		if run is None:
			target = self.avg_results()
		else:
			target = self.results[run]
		with open(savepath, 'w') as file:
			json.dump(target, file)

	def save_figures(self, savefolder='./images/', run=None):
		print("Exporting images to {}".format(savefolder))
		if run is None:
			target = self.avg_results()
		else:
			target = self.results[run]
		for detector_type, detector_result in target.items():
			fig_f1, ax_f1 = plt.subplots()
			ax_f1.plot(self.outlier_fractions, detector_result['Recall'], color='green', label='Recall')
			ax_f1.plot(self.outlier_fractions, detector_result['Precision'], color='blue', label='Precision')
			ax_f1.plot(self.outlier_fractions, detector_result['F1'], color='red', label='F1')
			ax_f1.legend(loc=0)
			fig_roc, ax_roc = plt.subplots()
			ax_roc.plot(detector_result['FP_rate'], detector_result['Recall'])
			fig_f1.savefig(os.path.join(savefolder, "{}_F1.png".format(detector_type).replace(' ', '_')), bbox_inches='tight', dpi=300)
			fig_roc.savefig(os.path.join(savefolder, "{}_ROC.png".format(detector_type).replace(' ', '_')), bbox_inches='tight', dpi=300)
