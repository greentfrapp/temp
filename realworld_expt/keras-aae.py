from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras.datasets import mnist
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE
from absl import flags
from absl import app
import json
from collections import Counter
from scipy.stats import chi


from utils import MammoData
from utils import CardioData
from utils import ThyroidData
from utils import LymphoData


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("adversarial", True, "Use Adversarial Autoencoder or regular Autoencoder")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("reconstruct", False, "Reconstruct image")
flags.DEFINE_bool("generate", False, "Generate image from latent")
flags.DEFINE_bool("generate_grid", False, "Generate grid of images from latent space (only for 2D latent)")
flags.DEFINE_bool("duplicate_edge", False, "Duplicate points at edge of data")
flags.DEFINE_bool("decode_edge", False, "Decode points at edge of data")
flags.DEFINE_bool("jitter", False, "Decode from jittered points at edge of data")
flags.DEFINE_bool("smote", False, "Sample with SMOTE")
flags.DEFINE_bool("plot", False, "Plot latent space")
flags.DEFINE_integer("latent_dim", 2, "Latent dimension")

# Train
flags.DEFINE_integer("epochs", 50, "Number of training epochs")
flags.DEFINE_integer("train_samples", 10000, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize", 100, "Training batchsize")

# Test
flags.DEFINE_integer("test_samples", 10000, "Number of test samples from MNIST")
flags.DEFINE_list("latent_vec", None, "Latent vector (use with --generate flag)")


def create_model(input_dim, latent_dim, verbose=False, save_graph=False):

	autoencoder_input = Input(shape=(input_dim,))
	generator_input = Input(shape=(input_dim,))

	encoder = Sequential()
	encoder.add(Dense(1000, input_shape=(input_dim,), activation='relu'))
	encoder.add(Dense(1000, activation='relu'))
	encoder.add(Dense(latent_dim, activation=None))
	
	decoder = Sequential()
	decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
	decoder.add(Dense(1000, activation='relu'))
	decoder.add(Dense(input_dim, activation='sigmoid'))

	if FLAGS.adversarial:
		discriminator = Sequential()
		discriminator.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
		discriminator.add(Dense(1000, activation='relu'))
		discriminator.add(Dense(1, activation='sigmoid'))

	autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
	autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")
	
	if FLAGS.adversarial:
		discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
		discriminator.trainable = False
		generator = Model(generator_input, discriminator(encoder(generator_input)))
		generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")

	if verbose:
		print("Autoencoder Architecture")
		print(autoencoder.summary())
		if FLAGS.adversarial:
			print("Discriminator Architecture")
			print(discriminator.summary())
			print("Generator Architecture")
			print(generator.summary())

	if save_graph:
		plot_model(autoencoder, to_file="autoencoder_graph.png")
		if FLAGS.adversarial:
			plot_model(discriminator, to_file="discriminator_graph.png")
			plot_model(generator, to_file="generator_graph.png")

	if FLAGS.adversarial:
		return autoencoder, discriminator, generator, encoder, decoder
	else:
		return autoencoder, None, None, encoder, decoder

def train(n_samples, batch_size, n_epochs):
	autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=21, latent_dim=FLAGS.latent_dim)
	# dataset = MammoData()
	dataset = CardioData()
	# dataset = ThyroidData()
	# dataset = LymphoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	# Get n_samples/10 samples from each class
	x = x_test
	y = y_test

	rand_x = np.random.RandomState(42)
	rand_y = np.random.RandomState(42)

	past = datetime.now()
	for epoch in np.arange(1, n_epochs + 1):
		autoencoder_losses = []
		if FLAGS.adversarial:
			discriminator_losses = []
			generator_losses = []
		rand_x.shuffle(x)
		rand_y.shuffle(y)
		for batch in np.arange(len(x) / batch_size):
			start = int(batch * batch_size)
			end = int(start + batch_size)
			samples = x[start:end]
			if end > len(x):
				samples = np.concatenate((samples, x[:end - len(x)]))
			autoencoder_history = autoencoder.fit(x=samples, y=samples, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			if FLAGS.adversarial:
				fake_latent = encoder.predict(samples)
				discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, FLAGS.latent_dim) * 10.))
				discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
				discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
				generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			
			autoencoder_losses.append(autoencoder_history.history["loss"])
			if FLAGS.adversarial:
				discriminator_losses.append(discriminator_history.history["loss"])
				generator_losses.append(generator_history.history["loss"])
		now = datetime.now()
		print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
		print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
		if FLAGS.adversarial:
			print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
			print("Generator Loss: {}".format(np.mean(generator_losses)))
		past = now

		if epoch % 50 == 0:
			print("\nSaving models...")
			encoder.save('{}_encoder_10_test.h5'.format(desc))
			decoder.save('{}_decoder_10_test.h5'.format(desc))
	encoder.save('{}_encoder_10_test.h5'.format(desc))
	decoder.save('{}_decoder_10_test.h5'.format(desc))

def plot(n_samples):
	encoder = load_model('{}_encoder_10_test.h5'.format(desc))
	dataset = MammoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_test
	y = y_test
	labels = ["Nominal", "Anomaly"]
	latent = encoder.predict(x)
	if FLAGS.latent_dim > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	fig, ax = plt.subplots()
	for label in np.arange(2):
		ax.scatter(latent[(y == label), 0], latent[(y == label), 1], label=labels[label], s=3)
	
	center = np.mean(latent, axis=0)
	std = np.std(latent, axis=0)
	chi_std = chi.std(2, 0, np.linalg.norm(std))
	dist = np.linalg.norm(latent - center, axis=1)
	for i, el in enumerate(dist):
		if el > 2.7 * chi_std:
			dist[i] = 0.
	sorted_dist = np.copy(dist)
	sorted_dist.sort()
	threshold = sorted_dist[int(len(dist) * 0.9)]
	for i, el in enumerate(dist):
		if el < threshold:
			dist[i] = 0
	dist /= np.sum(dist)
	samples = []
	choices = []
	for i in np.arange(1100):
		choice = np.random.choice(np.arange(len(dist)), p=dist)
		choices.append(choice)
		samples.append(latent[choice].tolist())
	print(Counter(choices))
	samples = np.array(samples)
	ax.scatter(samples[:,0], samples[:,1], label="duplicate", s=3, color='#e74c3c')

	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	ax.set_aspect('equal')
	ax.set_xlim(-40, 40)
	ax.set_ylim(-40, 40)
	ax.set_title("Latent Space")
	plt.show(block=False)
	raw_input("Press Enter to Exit")

def generate():
	decoder = load_model('{}_decoder.h5'.format(desc))
	latent = np.random.randn(1100, 2)
	for i, vector in enumerate(latent):
		latent[i] = 10. * vector / np.linalg.norm(vector)
	latent += [0, -5]
	samples = decoder.predict(latent.reshape(-1, FLAGS.latent_dim))
	with open("synthetic_samples.json", 'w') as file:
		json.dump(samples.tolist(), file)

def duplicate_edge(decode=False):
	encoder = load_model('{}_encoder.h5'.format(desc))
	decoder = load_model('{}_decoder.h5'.format(desc))
	dataset = MammoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	latent = encoder.predict(x)
	labels = ["Nominal", "Anomaly"]
	if FLAGS.latent_dim > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	center = np.mean(latent, axis=0)

	dist = np.linalg.norm(latent - center, axis=1)
	for i, el in enumerate(dist):
		if el > 20.:
			dist[i] = 0.
		elif el < 10:
			dist[i] = 0.
	dist /= np.sum(dist)

	samples = []
	for i in np.arange(1100):
		choice = np.random.choice(np.arange(len(dist)), p=dist)
		if decode:
			samples.append(latent[choice])
		else:
			samples.append(x_train[choice].tolist())
	if decode:
		samples = decoder.predict(latent.reshape(-1, FLAGS.latent_dim)).tolist()
	with open("duplicated_samples.json", 'w') as file:
		json.dump(samples, file)

def smote():
	encoder = load_model('{}_encoder.h5'.format(desc))
	decoder = load_model('{}_decoder.h5'.format(desc))
	dataset = MammoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	latent = encoder.predict(x)
	labels = ["Nominal", "Anomaly"]
	if FLAGS.latent_dim > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	center = np.mean(latent, axis=0)

	dist = np.linalg.norm(latent - center, axis=1)
	for i, el in enumerate(dist):
		if el > 20.:
			dist[i] = 0.
		elif el < 10:
			dist[i] = 0.
	dist /= np.sum(dist)

	synth_latent = []
	for i in np.arange(1100):
		choice = np.random.choice(np.arange(len(dist)), p=dist)
		a = latent[choice]
		latent_copy = np.concatenate((latent[:choice], latent[choice + 1:]))
		latent_copy -= a
		latent_copy = np.linalg.norm(latent_copy, axis=1)
		b = np.argmin(latent_copy)
		if b >= choice:
			b += 1
		b = latent[b]
		scale = np.random.rand()
		c = scale * (a-b) + b
		synth_latent.append(c)
	samples = decoder.predict(np.array(synth_latent).reshape(-1, FLAGS.latent_dim))
	with open("smote_samples.json", 'w') as file:
		json.dump(samples.tolist(), file)

def jitter():
	encoder = load_model('{}_encoder.h5'.format(desc))
	decoder = load_model('{}_decoder.h5'.format(desc))
	dataset = MammoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	x = x_train
	y = y_train
	latent = encoder.predict(x)
	labels = ["Nominal", "Anomaly"]
	if FLAGS.latent_dim > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	center = np.mean(latent, axis=0)

	dist = np.linalg.norm(latent - center, axis=1)
	for i, el in enumerate(dist):
		if el > 20.:
			dist[i] = 0.
		elif el < 10:
			dist[i] = 0.
	dist /= np.sum(dist)

	synth_latent = []
	for i in np.arange(1100):
		choice = np.random.choice(np.arange(len(dist)), p=dist)
		a = latent[choice]
		synth_latent.append(a + np.random.randn(*np.array(a).shape) * 0.01)
	samples = decoder.predict(np.array(synth_latent).reshape(-1, FLAGS.latent_dim))
	with open("jitter_samples.json", 'w') as file:
		json.dump(samples.tolist(), file)

def main(argv):
	global desc
	if FLAGS.adversarial:
		desc = "aae"
	else:
		desc = "regular"
	if FLAGS.train:
		train(n_samples=FLAGS.train_samples, batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs)
	elif FLAGS.reconstruct:
		reconstruct(n_samples=FLAGS.test_samples)
	elif FLAGS.generate:
		if FLAGS.latent_vec:
			assert len(FLAGS.latent_vec) == FLAGS.latent_dim, "Latent vector provided is of dim {}; required dim is {}".format(len(FLAGS.latent_vec), FLAGS.latent_dim)
			generate(FLAGS.latent_vec)
		else:
			generate()
	elif FLAGS.generate_grid:
		generate_grid()
	elif FLAGS.duplicate_edge:
		duplicate_edge()
	elif FLAGS.decode_edge:
		duplicate_edge(decode=True)
	elif FLAGS.smote:
		smote()
	elif FLAGS.jitter:
		jitter()
	elif FLAGS.plot:
		plot(FLAGS.test_samples)


if __name__ == "__main__":
	app.run(main)
