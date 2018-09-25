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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import numpy as np
from matplotlib import rcParams

rcParams.update({'font.size': 10})

def main():
	generators = []
	generators.append(GeneratorToy1())
	generators.append(GeneratorToy2())
	#generators.append(GeneratorToy3())
	#generators.append(GeneratorToy4())
	#generators.append(GeneratorToy5())
	#generators.append(GeneratorToy6())
	#generators.append(GeneratorToy7())
	#generators.append(GeneratorToy8())
	generators.append(GeneratorToy10())
	anomalous_datasets = []
	nominal_datasets = []
	for generator in generators:
		dataset = generator.generate()
		anomalous_datasets.append([])
		nominal_datasets.append([])
		for idx, value in enumerate(dataset.values):
			if dataset.anomaly_labels[idx] == 1:
				nominal_datasets[-1].append(value)
			if dataset.anomaly_labels[idx] == -1:
				anomalous_datasets[-1].append(value)

	fig = plt.figure(figsize=(21, 4))
	for i, _ in enumerate(nominal_datasets):
		if len(nominal_datasets[i][0]) == 2:
			ax = fig.add_subplot(1, 3, i + 1)
			ax.scatter(np.array(nominal_datasets[i])[:, 0], np.array(nominal_datasets[i])[:, 1], color='#e74c3c', label='Nominal', s=3)
			ax.scatter(np.array(anomalous_datasets[i])[:, 0], np.array(anomalous_datasets[i])[:, 1], color='#2c3e50', label='Anomalous', s=3)
		elif len(nominal_datasets[i][0]) == 3:
			ax = fig.add_subplot(1, 3, i + 1, projection='3d')
			if i == 1 or i == 2:
				ax.scatter(np.array(nominal_datasets[i])[:, 0], np.array(nominal_datasets[i])[:, 1], np.array(nominal_datasets[i])[:, 2], color='#e74c3c', label='Nominal', s=3)
			else:
				ax.scatter(np.array(nominal_datasets[i])[:, 0], np.array(nominal_datasets[i])[:, 1], np.array(nominal_datasets[i])[:, 2], color='#e74c3c', label='Nominal', s=3)
			ax.scatter(np.array(anomalous_datasets[i])[:, 0], np.array(anomalous_datasets[i])[:, 1], np.array(anomalous_datasets[i])[:, 2], color='#2c3e50', label='Anomalous', s=3)
			#ax.set_zticks([])
		ax.set_aspect('equal')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
		if i == 0:
			ax.set_title('Dataset A', fontsize=14)
		if i == 1:
			ax.set_title('Dataset B', fontsize=14)
		if i == 2:
			ax.set_title('Dataset C', fontsize=14)
		#ax.set_xticks([])
		#ax.set_yticks([])

	fig.savefig('original_dataspace2.png', bbox_inches='tight', dpi=300)
	
	#plt.show(block=False)
	#raw_input('ENTER')

if __name__ == "__main__":
	main()