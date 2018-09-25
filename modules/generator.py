"""
generator.py
Generates datasets for anomaly detectors and AAEs.
Datasets can be saved as .json files.
"""

import argparse
import numpy as np
from numpy.random import RandomState
import json

# HELPER FUNCTIONS

def combine(a, b):
	if len(a) == 0:
		return b
	else:
		return np.concatenate((a, b))

def shuffle(data, random_seed=1):
	data_copy = np.copy(data).tolist()
	rand = RandomState(random_seed)
	rand.shuffle(data_copy)
	return data_copy

# GENERATOR CLASS

class GeneratorToy1(object):

	def __init__(self, contamination=0.05):

		self.contamination = contamination

	@property
	def contamination(self):
		return self._contamination

	@contamination.setter
	def contamination(self, value):
		assert isinstance(value, (int, float)) and value <= 1.
		self._contamination = value

	# generates a list of dicts where each dict has 'value' and 'label' keys
	# 'value' is the sample vector
	# 'label' is -1 for anomalous data and 1 for nominal data
	# fully reproducible ie. calling generate() with the same random_seed and size will always give the same result
	def generate(self, size=1000, savepath=None, random_seed=1, test=False):

		nominal_datapoints = []
		anomalous_datapoints = []

		anomalous_size = int(self.contamination * size)
		nominal_size = size - anomalous_size

		print("Begin generating {} total datapoint(s) - {} nominal, {} anomalous...".format(size, nominal_size, anomalous_size))

		nominal_size = int(size * (1. - self.contamination))
		anomaly_size = size - nominal_size

		nominal_datapoints = self.generate_nominal(nominal_size, random_seed)
		anomalous_datapoints = self.generate_anomaly(anomaly_size, random_seed)

		datapoints = []
		for point in nominal_datapoints:
			datapoint = dict()
			datapoint['value'] = point
			datapoint['anomaly_label'] = 1
			datapoints.append(datapoint)
		for point in anomalous_datapoints:
			datapoint = dict()
			datapoint['value'] = point
			datapoint['anomaly_label'] = -1
			datapoints.append(datapoint)

		datapoints = shuffle(datapoints, random_seed=random_seed)

		output = Dataset()
		for datapoint in datapoints:
			output.values.append(datapoint['value'])
			output.anomaly_labels.append(datapoint['anomaly_label'])

		print("Complete generating {} total datapoint(s) - {} nominal, {} anomalous...".format(size, nominal_size, anomalous_size))

		if savepath is not None:
			print("Saving datapoints to {}".format(savepath))
			output.save(savepath)

		return output

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 2)

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 2) + np.array([[10., 0.]])

class GeneratorToy2(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 3)

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 3) + np.array([[10., 0., 0.]])

class GeneratorToy3(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		output = []
		for i in np.arange(size):
			radius = rand.randn(1)[0] + 10.
			random_vector = rand.randn(2)
			random_vector /= np.linalg.norm(random_vector) / radius
			output.append(random_vector)
		return output

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 2)

class GeneratorToy4(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		output = []
		for i in np.arange(size):
			radius = rand.randn(1)[0] + 10.
			random_vector = rand.randn(3)
			random_vector /= np.linalg.norm(random_vector) / radius
			output.append(random_vector)
		return output

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(size, 3)

class GeneratorToy5(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 2)) * 2. - 1.

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 2)) * 0.5 + np.array([[5., -0.25]])

class GeneratorToy6(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 3)) * 2. - 1.

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 3)) * 0.5 + np.array([[5., -0.25, -0.25]])

class GeneratorToy7(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		output = []
		for i in np.arange(size):
			if rand.random_sample(1)[0] < 0.6:
				if rand.random_sample(1)[0] < 0.5:
					random_vector = rand.random_sample((2, )) * np.array([2., 12.]) + np.array([4., -6.])
				else:
					random_vector = rand.random_sample((2, )) * np.array([2., 12.]) + np.array([-6., -6.])
			else:
				if rand.random_sample(1)[0] < 0.5:
					random_vector = rand.random_sample((2, )) * np.array([8., 2.]) + np.array([-4., -6.])
				else:
					random_vector = rand.random_sample((2, )) * np.array([8., 2.]) + np.array([-4., 4.])
			output.append(random_vector)
		return output

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 2)) * 2. + np.array([[-1., -1.]])

class GeneratorToy8(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		output = []
		for i in np.arange(size):
			choices = np.concatenate((np.zeros(72), np.ones(48), np.ones(32) * 2))
			choice = rand.choice(choices)
			if choice == 0:
				if rand.random_sample(1)[0] < 0.5:
					random_vector = rand.random_sample((3, )) * np.array([2., 12., 12.]) + np.array([4., -6., -6.])
				else:
					random_vector = rand.random_sample((3, )) * np.array([2., 12., 12.]) + np.array([-6., -6., -6.])
			elif choice == 1:
				if rand.random_sample(1)[0] < 0.5:
					random_vector = rand.random_sample((3, )) * np.array([8., 2., 12.]) + np.array([-4., -6., -6.])
				else:
					random_vector = rand.random_sample((3, )) * np.array([8., 2., 12.]) + np.array([-4., 4., -6.])
			else:
				if rand.random_sample(1)[0] < 0.5:
					random_vector = rand.random_sample((3, )) * np.array([8., 8., 2.]) + np.array([-4., -4., -6.])
				else:
					random_vector = rand.random_sample((3, )) * np.array([8., 8., 2.]) + np.array([-4., -4., 4.])
			output.append(random_vector)
		return output

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.random_sample((size, 3)) * 2. + np.array([[-1., -1., -1.]])

class GeneratorToy10(GeneratorToy1):

	def generate_nominal(self, size, random_seed):
		rand = RandomState(random_seed)
		output = []
		for i in np.arange(size):
			radius = rand.randn(1)[0] * 5. + 30.
			sample = rand.randn(2)
			sample /= np.linalg.norm(sample) / radius
			output.append(sample)
		return output

	def generate_anomaly(self, size, random_seed):
		rand = RandomState(random_seed)
		return rand.randn(50, 2) * 5.

class Dataset(object):

	def __init__(self):
		self.values = []
		self.anomaly_labels = []
		self.is_test = []
		self.idx = 0

	def load(self, savepath):
		with open(savepath, 'r') as file:
			dataset = json.load(file)
		self.values = dataset['values']
		self.anomaly_labels = dataset['anomaly_labels']

	def save(self, savepath):
		dataset = dict()
		dataset['values'] = self.values
		dataset['anomaly_labels'] = self.anomaly_labels
		with open(savepath, 'w') as file:
			json.dump(dataset, file)

	def next_batch(self, batch_size, shuffle=False):
		start = self.idx
		end = self.idx + batch_size
		if end == self.num_examples:
			self.idx = 0
			return self.values[start:], self.anomaly_labels[start:]
		if end > self.num_examples:
			end -= self.num_examples
			self.idx = end
			values = np.concatenate((self.values[start:], self.values[:end])).tolist()
			anomaly_labels = np.concatenate((self.anomaly_labels[start:], self.anomaly_labels[:end])).tolist()
			return values, anomaly_labels
		else:
			self.idx = end
			return self.values[start:end], self.anomaly_labels[start:end]

	def merge_with(self, dataset_b, shuffle=False):
		merged_dataset = Dataset()
		merged_dataset.values = np.copy(self.values).tolist()
		merged_dataset.anomaly_labels = np.copy(self.anomaly_labels).tolist()
		merged_dataset.is_test = np.copy(self.is_test).tolist()
		for idx, value in enumerate(dataset_b.values):
			merged_dataset.values.append(value)
			merged_dataset.anomaly_labels.append(dataset_b.anomaly_labels[idx])
			merged_dataset.is_test.append(dataset_b.is_test[idx])
		return merged_dataset

	@property
	def num_examples(self):
		return len(self.values)

	@property
	def values(self):
		return self._values

	@values.setter
	def values(self, value):
		self._values = value

	@property
	def anomaly_labels(self):
		return self._anomaly_labels

	@anomaly_labels.setter
	def anomaly_labels(self, value):
		self._anomaly_labels = value

	@property
	def is_test(self):
		return self._is_test

	@is_test.setter
	def is_test(self, value):
		self._is_test = value

def main():
	return None

if __name__ == "__main__":
	main()

