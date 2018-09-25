import scipy.io
import numpy as np


class MammoData(object):

	def __init__(self):
		super(MammoData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("realworlddata/mammography.mat")
		samples = data['X'] # 11183 samples
		labels = data['y']

		nominal_samples = samples[np.where(labels[:, 0] == 0)] # 10923 nominal
		nominal_labels = labels[np.where(labels[:, 0] == 0)]
		anomalous_samples = samples[np.where(labels[:, 0] == 1)] # 260 anomalies
		anomalous_labels = labels[np.where(labels[:, 0] == 1)]

		n_train = int(len(nominal_samples) / 2)
		n_anomalous_test = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train, :] # 5461 training
		y_train = nominal_labels[:n_train].reshape(-1)

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_anomalous_test:])) # 5722 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_anomalous_test:])).reshape(-1)

		return (x_train, y_train), (x_test, y_test)

class CardioData(object):

	def __init__(self):
		super(CardioData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("realworlddata/cardio.mat")
		samples = data['X'] # 1831 samples
		labels = data['y']

		# normalize
		data_max = np.max(samples, axis=0)
		data_min = np.min(samples, axis=0)
		samples -= data_min
		samples /= data_max - data_min

		nominal_samples = samples[np.where(labels[:, 0] == 0)] # 1655 nominal
		nominal_labels = labels[np.where(labels[:, 0] == 0)]
		anomalous_samples = samples[np.where(labels[:, 0] == 1)] # 176 anomalies
		anomalous_labels = labels[np.where(labels[:, 0] == 1)]

		n_train = int(len(nominal_samples) / 2)
		n_test_anomalous = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train, :] # 827 training
		y_train = nominal_labels[:n_train].reshape(-1)

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_test_anomalous:])) # 1005 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_test_anomalous:])).reshape(-1)

		return (x_train, y_train), (x_test, y_test)

class ThyroidData(object):

	def __init__(self):
		super(ThyroidData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("realworlddata/thyroid.mat")
		samples = data['X'] # 3772 samples
		labels = data['y']

		nominal_samples = samples[np.where(labels[:, 0] == 0)] # 3679 nominal
		nominal_labels = labels[np.where(labels[:, 0] == 0)]
		anomalous_samples = samples[np.where(labels[:, 0] == 1)] # 93 anomalies
		anomalous_labels = labels[np.where(labels[:, 0] == 1)]

		n_train = int(len(nominal_samples) / 2)
		n_anomalous_test = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train] # 1839 training
		y_train = nominal_labels[:n_train].reshape(-1)

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_anomalous_test:])) # 1933 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_anomalous_test:])).reshape(-1)

		return (x_train, y_train), (x_test, y_test)

class LymphoData(object):

	def __init__(self):
		super(LymphoData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("realworlddata/lympho.mat")
		samples = data['X'] # 148 samples
		labels = data['y']

		# normalize
		data_max = np.max(samples, axis=0)
		data_min = np.min(samples, axis=0)
		samples -= data_min
		samples /= data_max - data_min

		nominal_samples = samples[np.where(labels[:, 0] == 0)] # 142 nominal
		nominal_labels = labels[np.where(labels[:, 0] == 0)]
		anomalous_samples = samples[np.where(labels[:, 0] == 1)] # 6 anomalies
		anomalous_labels = labels[np.where(labels[:, 0] == 1)]

		n_train = int(len(nominal_samples) / 2)
		n_test_anomalous = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train, :] # 71 training
		y_train = nominal_labels[:n_train].reshape(-1)

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_test_anomalous:])) # 77 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_test_anomalous:])).reshape(-1)

		return (x_train, y_train), (x_test, y_test)