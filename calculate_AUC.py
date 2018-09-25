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

rcParams.update({'font.size': 14})

def calculate_AUC():
	detector = "Isolation Forest"
	detector = "One-class SVM"
	#"""
	with open('Toy10_Labeled_results/original.json', 'r') as file:
		original_results = json.load(file)
	if detector == "Robust Deep Autoencoder":
		original_results[detector]["Recall"].reverse()
		original_results[detector]["FP_rate"].reverse()
	y = np.concatenate([[0,0],[0,0], original_results[detector]["Recall"], [1,1]])
	x = np.concatenate([[0,0],[0,0], original_results[detector]["FP_rate"], [1,1]])
	original_auc = np.trapz(y, x)
	#"""
	
	folder = 'Toy10_Labeled_results/+100'
	
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

	auc = auc[:]
	dists = dists[:]
	fig, ax = plt.subplots()
	ax.plot(dists, auc, color='#27ae60', label='Augmented Data', linewidth=3.)
	ax.plot(dists, np.ones_like(auc) * original_auc, linestyle='--', color='#2c3e50', label='Original Data', linewidth=3.)
	ax.set_xlim(0, dists[-1])
	ax.text(.5,1.07,'Area-Under-Curve (AUC) against Magnitude', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	if detector == "One-class SVM":
		ax.text(.5,1.02,'One-class SVM', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	elif detector == "Isolation Forest":
		ax.text(.5,1.02,'Isolation Forest', fontsize=16, horizontalalignment='center', transform=ax.transAxes)
	ax.legend(loc=0, frameon=True, fontsize=14)
	ax.set_xlabel('Magnitude')
	ax.set_ylabel('AUC')
	if detector == "One-class SVM":
		fig.savefig(os.path.join(folder, '../images', 'AUC_SVM_+100.png'), bbox_inches='tight', dpi=300)
	elif detector == "Isolation Forest":
		fig.savefig(os.path.join(folder, '../images', 'AUC_IF_+100.png'), bbox_inches='tight', dpi=300)

	print dists[np.argmax(auc)]
	

if __name__ == "__main__":
	calculate_AUC()
