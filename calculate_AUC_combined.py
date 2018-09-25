from modules.aae_util import BasicAdversarialAutoencoder
from modules.aae_labeled import LabeledAdversarialAutoencoder
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import os
import numpy as np
import datetime
import json
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})

def calculate_AUC():
	detector = "Isolation Forest"
	#detector = "One-class SVM"

	result_folders = ['Toy1_Labeled*_results', 'Toy2_Labeled*_results', 'Toy10_Labeled*_results']
	dataset_names = ['A', 'B', 'C']
	original_auc = []
	doping_auc = []

	for folder in result_folders:
		with open(folder + '/original.json', 'r') as file:
			original_results = json.load(file)
		y = np.concatenate([[0,0],[0,0], original_results[detector]["Recall"], [1,1]])
		x = np.concatenate([[0,0],[0,0], original_results[detector]["FP_rate"], [1,1]])
		original_auc.append(np.trapz(y, x))

		folder = folder + '/+100'
	
		auc_dists = []

		for filename in os.listdir(folder)[:]:
			if filename != '.DS_Store' and 'json' in filename:
				with open(os.path.join(folder, filename), 'r') as file:
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

		doping_auc.append([dists, auc])

	fig = plt.figure(figsize=(10, 31))
	for i, _ in enumerate(result_folders):
		ax = fig.add_subplot(3, 1, i + 1)
		ax.plot(doping_auc[i][0], doping_auc[i][1], color='#27ae60', label='Augmented Data', linewidth=3.)
		ax.plot(doping_auc[i][0], np.ones_like(doping_auc[i][1]) * original_auc[i], linestyle='--', color='#2c3e50', label='Original Data', linewidth=3.)
		ax.set_xlim(0, dists[-1])

		ax.text(.5,1.02,'Dataset {}: AUC against Magnitude'.format(dataset_names[i]), fontsize=20, horizontalalignment='center', transform=ax.transAxes)
		#if detector == "One-class SVM":
		#	ax.text(.5,1.02,'One-class SVM', fontsize=20, horizontalalignment='center', transform=ax.transAxes)
		#elif detector == "Isolation Forest":
		#	ax.text(.5,1.02,'Isolation Forest', fontsize=20, horizontalalignment='center', transform=ax.transAxes)
		ax.legend(loc=0, frameon=True, fontsize=18)
		ax.set_xlabel('Magnitude')
		ax.set_ylabel('AUC')

	if detector == "One-class SVM":
		fig.savefig('AUC_SVM_+100.png', bbox_inches='tight', dpi=300)
	elif detector == "Isolation Forest":
		fig.savefig('AUC_IF_+100.png', bbox_inches='tight', dpi=300)
	

if __name__ == "__main__":
	calculate_AUC()
