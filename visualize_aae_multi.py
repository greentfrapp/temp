from modules.aae_labeled import LabeledAdversarialAutoencoder
from modules.aae_util import BasicAdversarialAutoencoder
from modules.generator import GeneratorToy1
from modules.generator import GeneratorToy2
from modules.generator import GeneratorToy3
from modules.generator import GeneratorToy4
from modules.generator import GeneratorToy5
from modules.generator import GeneratorToy6
from modules.generator import GeneratorToy7
from modules.generator import GeneratorToy8
from modules.generator import GeneratorToy10
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

	models = ['Toy1_Labeled', 'Toy2_Labeled', 'Toy10_Labeled']
	generators = [GeneratorToy1(), GeneratorToy2(), GeneratorToy10()]
	datasets = []
	for generator in generators:
		datasets.append(generator.generate())

	aae = LabeledAdversarialAutoencoder(data_dim=2, z_dim=2)
	aae.load(os.path.join('models', model))
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
	ax.text(.5,1.07,'Latent Space of Toy Experiment 10', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	ax.legend(loc=3, frameon=True, fontsize=14)
	ax.axis('equal')
	#ax.set_xlim(-60, 60)
	#ax.set_ylim(-60, 60)
	# plt.show(block=False)
	# raw_input("Enter")
	fig.savefig(os.path.join('models', model, 'latent_space.png'), bbox_inches='tight', dpi=300)