from modules.aae_labeled import LabeledAdversarialAutoencoder
from modules.aae_util import BasicAdversarialAutoencoder
from modules.generator import GeneratorToy1
from modules.generator import GeneratorToy2
from modules.generator import GeneratorToy10
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
import json
from scipy.integrate import simps
from matplotlib import rcParams
rcParams.update({'font.size': 8})

def main():
	aae = LabeledAdversarialAutoencoder(data_dim=2, z_dim=2)
	model_path = './models/Toy10_Labeled'
	aae.load(model_path)

	generator = GeneratorToy10()
	real_samples = generator.generate()

	dists = [2.0, 10.0, 100.0]
	colors = ['#2ecc71', '#3498db', '#9b59b6']

	fig = plt.figure()
	gs = gridspec.GridSpec(3, 1, wspace=0.4, hspace=0.5)
	ax = [[],[]]
	ax[0].append(fig.add_subplot(gs[0,0]))
	ax[1].append(fig.add_subplot(gs[1:,0]))
	
	nominal_real = {'x': [], 'y': []}
	anomalous_real = {'x': [], 'y': []}
	nominal_latent = {'x': [], 'y': []}
	anomalous_latent = {'x': [], 'y': []}
	for idx, value in enumerate(real_samples.values):
		latent_vector = aae.encode(value)
		if real_samples.anomaly_labels[idx] == -1:
			anomalous_real['x'].append(value[0])
			anomalous_real['y'].append(value[1])
			anomalous_latent['x'].append(latent_vector[0])
			anomalous_latent['y'].append(latent_vector[1])
		elif real_samples.anomaly_labels[idx] == 1:
			nominal_real['x'].append(value[0])
			nominal_real['y'].append(value[1])
			nominal_latent['x'].append(latent_vector[0])
			nominal_latent['y'].append(latent_vector[1])

	ax[0][0].scatter(nominal_real['x'], nominal_real['y'], color='#e74c3c', label='Nominal', s=3)
	ax[0][0].scatter(anomalous_real['x'], anomalous_real['y'], color='#2c3e50', label='Anomalous', s=3)
	ax[1][0].scatter(nominal_latent['x'], nominal_latent['y'], color='#e74c3c', label='Nominal', s=3)
	ax[1][0].scatter(anomalous_latent['x'], anomalous_latent['y'], color='#2c3e50', label='Anomalous', s=3)

	for idx, dist in enumerate(dists):
		vectors, samples = aae.generate(mode='distribution', size=100, distribution=dist)
		ax[0][0].scatter(np.array(samples).T[0], np.array(samples).T[1], color=colors[idx], label='Mag{}'.format(dist), s=3)
		ax[1][0].scatter(np.array(vectors).T[0], np.array(vectors).T[1], color=colors[idx], label='Mag{}'.format(dist), s=3)
		
	ax[0][0].text(.5,1.27,'Dataset C', fontsize=12, horizontalalignment='center', transform=ax[0][0].transAxes)
	ax[0][0].text(.5,1.07,'Original Data Space', fontsize=10, horizontalalignment='center', transform=ax[0][0].transAxes)
	ax[0][0].set_aspect('equal')
	#ax[0][0].legend(loc=0, frameon=True, fontsize=12)
	#ax[0][0].set_xlim(-150, 350)
	#ax[0][0].set_ylim(-80, 80)
	ax[0][0].set_xlim(-200, 200)
	ax[0][0].set_ylim(-80, 80)

	ax[1][0].text(.5,1.07,'Latent Space', fontsize=10, horizontalalignment='center', transform=ax[1][0].transAxes)
	ax[1][0].set_aspect('equal')
	box = ax[1][0].get_position()
	ax[1][0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax[1][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10, columnspacing=1)
	#ax[1][0].legend(loc=0, frameon=True, fontsize=12)
	ax[1][0].set_ylim(-150, 150)
	ax[1][0].set_xlim(-150, 150)
	
	fig.savefig('DatasetC_mapping2.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
	main()
