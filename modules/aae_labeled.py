from aae_util import BasicAdversarialAutoencoder
from aae_util import DenseNetwork
import tensorflow as tf
from numpy.random import RandomState
import numpy as np
import os


class LabeledAdversarialAutoencoder(BasicAdversarialAutoencoder):

	def __init__(self, data_dim=2, z_dim=2, batch_size=100, n_classes=2):

		# Parameters for everything
		self.data_dim = data_dim
		self.z_dim = z_dim
		self.batch_size = batch_size
		self.real_prior_mean = 0.0
		self.real_prior_stdev = 5.0
		self.learning_rate = 0.001
		self.n_classes = n_classes
		if self.n_classes == 2:
			self.n_classes = 1

		# Initialize networks
		self.encoder = DenseNetwork(
			nodes_per_layer=[1000, 1000, self.z_dim],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["encoder_dense_1", "encoder_dense_2", "encoder_output"],
			network_name="Encoder")
		self.decoder = DenseNetwork(
			nodes_per_layer=[1000, 1000, self.data_dim],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["decoder_dense_1", "decoder_dense_2", "decoder_output"],
			network_name="Decoder")
		self.discriminator = DenseNetwork(
			nodes_per_layer=[1000, 1000, 1],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["discriminator_dense_1", "discriminator_dense_2", "discriminator_output"],
			network_name="Discriminator")

		# Create tf.placeholder variables for inputs
		self.original_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.data_dim], name='original_input')
		self.target_output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.data_dim], name='target_output')
		self.real_prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim], name='real_prior')
		self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_classes], name='onehot_label')
		self.sample_input = tf.placeholder(dtype=tf.float32, shape=[1, self.data_dim], name='sample_input')
		self.sample_latent_vector_input = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim], name='sample_latent_vector')

		# Outputs from forwardproping networks 
		with tf.variable_scope(tf.get_variable_scope()):
			self.latent_vector = self.encoder.forwardprop(self.original_input)
			self.reconstruction = self.decoder.forwardprop(self.latent_vector)
			self.score_real_prior = self.discriminator.forwardprop(tf.concat([self.real_prior, self.labels], axis=1))
			self.score_fake_prior = self.discriminator.forwardprop(tf.concat([self.latent_vector, self.labels], axis=1), reuse_variables=True)
			self.sample_latent_vector_output = self.encoder.forwardprop(self.sample_input, reuse_variables=True)
			self.sample_output = self.decoder.forwardprop(self.sample_latent_vector_input, reuse_variables=True)

		# Loss
		self.reconstruction_loss = tf.reduce_mean(tf.square(self.target_output - self.reconstruction))
		score_real_prior_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_real_prior), logits=self.score_real_prior))
		score_fake_prior_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.score_fake_prior), logits=self.score_fake_prior))
		self.discriminator_loss = score_real_prior_loss + score_fake_prior_loss
		self.encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_fake_prior), logits=self.score_fake_prior))

		# Filtering the variables to be trained
		all_variables = tf.trainable_variables()
		self.discriminator_variables = [var for var in all_variables if 'discriminator' in var.name]
		self.encoder_variables = [var for var in all_variables if 'encoder' in var.name]

		# Training functions
		self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.reconstruction_loss)
		self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.discriminator_loss, var_list=self.discriminator_variables)
		self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.encoder_loss, var_list=self.encoder_variables)
		
		# Things to save in Tensorboard
		tf.summary.scalar(name="Autoencoder Loss", tensor=self.reconstruction_loss)
		tf.summary.scalar(name="Discriminator Loss", tensor=self.discriminator_loss)
		tf.summary.scalar(name="Encoder Loss", tensor=self.encoder_loss)
		tf.summary.histogram(name="Encoder Distribution", values=self.latent_vector)
		tf.summary.histogram(name="Real Distribution", values=self.real_prior)
		self.summary_op = tf.summary.merge_all()

		# Boilerplate Tensorflow stuff
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

		return None

	def get_loss(self, batch_x, batch_y, z_real_dist):
		a_loss, d_loss, e_loss, summary = self.sess.run([self.reconstruction_loss, self.discriminator_loss, self.encoder_loss, self.summary_op], feed_dict={self.original_input:batch_x, self.target_output:batch_x, self.real_prior:z_real_dist, self.labels:np.array(batch_y).reshape(len(batch_y), self.n_classes)})
		return (a_loss, d_loss, e_loss, summary)

	def train(self, data=None, n_epochs=1000, results_folder='./Results'):

		# Create results_folder
		self.results_path = results_folder + '/AdversarialAutoencoder'
		if not os.path.exists(self.results_path):
			if not os.path.exists(results_folder):
				os.mkdir(results_folder)
			os.mkdir(self.results_path)

		self.n_epochs = n_epochs

		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.create_checkpoint_folders(self.z_dim, self.batch_size, self.n_epochs)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		if data is None:
			print("No data provided.")
			quit()

		for epoch in range(1, self.n_epochs + 1):
			n_batches = int(data.num_examples / self.batch_size)
			print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

			for batch in range(1, n_batches + 1):
				batch_x, batch_labels = data.next_batch(self.batch_size)
				z_real_dist = self.sample_prior(batch_labels)
				batch_labels = np.array(np.array(batch_labels) == 1, dtype=np.int32)
				
				autoencoder_learning_rate = 0.001
				discriminator_learning_rate = 0.001
				encoder_learning_rate = 0.001

				self.sess.run(self.autoencoder_optimizer, feed_dict={self.original_input:batch_x, self.target_output:batch_x})
				self.sess.run(self.discriminator_optimizer, feed_dict={self.original_input: batch_x, self.target_output: batch_x, self.real_prior: z_real_dist, self.labels:np.array(batch_labels).reshape(len(batch_labels), self.n_classes)})
				self.sess.run(self.encoder_optimizer, feed_dict={self.original_input: batch_x, self.target_output: batch_x, self.labels:np.array(batch_labels).reshape(len(batch_labels), self.n_classes)})

				# Print log and write to log.txt every 50 batches
				if batch % 10 == 0:
					a_loss, d_loss, e_loss, summary = self.get_loss(batch_x, batch_labels, z_real_dist)
					self.writer.add_summary(summary, global_step=self.step)
					print("Epoch: {}, iteration: {}".format(epoch, batch))
					print("Autoencoder Loss: {}".format(a_loss))
					print("Discriminator Loss: {}".format(d_loss))
					print("Generator Loss: {}".format(e_loss))
					with open(self.log_path + '/log.txt', 'a') as log:
						log.write("Epoch: {}, iteration: {}\n".format(epoch, batch))
						log.write("Autoencoder Loss: {}\n".format(a_loss))
						log.write("Discriminator Loss: {}\n".format(d_loss))
						log.write("Generator Loss: {}\n".format(e_loss))

				self.step += 1

			self.saver.save(self.sess, save_path=self.saved_model_path, global_step=self.step)

		print("Model Trained!")
		print("Tensorboard Path: {}".format(self.tensorboard_path))
		print("Log Path: {}".format(self.log_path + '/log.txt'))
		print("Saved Model Path: {}".format(self.saved_model_path))
		return None

	def sample_prior(self, labels):
		priors = []
		for label in labels:
			if label == 1:
				stdev = 5.
				mean = 0.
				prior = np.random.randn(1, self.z_dim) * stdev + mean
			else:
				radius = 100.
				prior = np.random.randn(1, self.z_dim)
				prior /= np.linalg.norm(prior) / radius
			if len(priors) == 0:
				priors = prior
			else:
				priors = np.concatenate((priors, prior))
		return priors

	def sample_latent_space(self, distribution=None, size=50, random_seed=1):
		#rand = RandomState(random_seed)
		vectors = np.random.randn(size, self.z_dim)
		for idx, vector in enumerate(vectors):
			vectors[idx] = float(distribution) * vector / np.linalg.norm(vector)
		return vectors

