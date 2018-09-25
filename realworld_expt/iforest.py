import scipy.io
import json
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from absl import flags
from absl import app


from utils import MammoData


FLAGS = flags.FLAGS

flags.DEFINE_bool("duplicate_edge", False, "Duplicate points at edge of data")
flags.DEFINE_bool("decode_edge", False, "Decode points at edge of data")
flags.DEFINE_bool("jitter", False, "Decode from jittered points at edge of data")
flags.DEFINE_bool("smote", False, "Sample with SMOTE")

def roc_val(classifier, x_test, y_test):

	predictions = classifier.predict(x_test)

	predicted_anomalies = (predictions == -1).astype(np.int32)

	tp = np.sum(predicted_anomalies[np.where(y_test == predicted_anomalies)] == 1)
	tn = np.sum(predicted_anomalies[np.where(y_test == predicted_anomalies)] == 0)
	fp = np.sum(predicted_anomalies) - tp
	fn = np.sum(predicted_anomalies == 0) - tn
	# print(tp, tn, fp, fn)

	if tp == 0:
		recall = tp_rate = 0.
		precision = 1.
	else:
		recall = tp_rate = tp / (tp + fn)
		precision = tp / (tp + fp)
	if recall + precision == 0:
		f1 = 0.
	else:
		f1 = (2 * recall * precision) / (recall + precision)
	fp_rate = fp / (fp + tn)
	# print(precision)
	# print(recall)
	# print(f1)

	return tp_rate, fp_rate, f1


def main(unused_argv):

	global desc
	if FLAGS.duplicate_edge:
		desc = "duplicated"
	elif FLAGS.decode_edge:
		desc = "duplicated"
	elif FLAGS.jitter:
		desc = "jitter"
	elif FLAGS.smote:
		desc = "smote"
	else:
		desc = "synthetic"

	dataset = MammoData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()

	print("Opening {}_samples.json".format(desc))
	with open("{}_samples.json".format(desc), 'r') as file:
		x_synth = np.array(json.load(file))
	x_synth = np.concatenate((x_train, x_synth))

	classifier_stats = {
		"TP rate": [],
		"FP rate": [],
		"F1": [],
	}
	classifier_synth_stats = {
		"TP rate": [],
		"FP rate": [],
		"F1": [],
	}

	con_vals = np.arange(0.01, 0.3, 0.02)
	con_vals = np.concatenate(([0.001], [0.005], con_vals))

	for i, con_val in enumerate(con_vals):

		print("Run #{}/{}".format(i + 1, len(con_vals)))

		orig =  {
			"TP rate": [],
			"FP rate": [],
			"F1": [],
		}
		synth = {
			"TP rate": [],
			"FP rate": [],
			"F1": [],
		}
		for j in np.arange(10):

			classifier = IsolationForest(contamination=con_val)
			classifier.fit(x_train)

			classifier_synth = IsolationForest(contamination=con_val)
			classifier_synth.fit(x_synth)

			tp_rate, fp_rate, f1 = roc_val(classifier, x_test, y_test)
			orig["TP rate"].append(tp_rate)
			orig["FP rate"].append(fp_rate)
			orig["F1"].append(f1)
			tp_rate, fp_rate, f1 = roc_val(classifier_synth, x_test, y_test)
			synth["TP rate"].append(tp_rate)
			synth["FP rate"].append(fp_rate)
			synth["F1"].append(f1)

		classifier_stats["TP rate"].append(np.mean(orig["TP rate"]))
		classifier_stats["FP rate"].append(np.mean(orig["FP rate"]))
		classifier_stats["F1"].append(np.mean(orig["F1"]))
		classifier_synth_stats["TP rate"].append(np.mean(synth["TP rate"]))
		classifier_synth_stats["FP rate"].append(np.mean(synth["FP rate"]))
		classifier_synth_stats["F1"].append(np.mean(synth["F1"]))

	print("Best original F1  - [{}]: {}".format(np.argmax(classifier_stats["F1"]), np.max(classifier_stats["F1"])))
	print("Best synthetic F1 - [{}]: {}".format(np.argmax(classifier_synth_stats["F1"]), np.max(classifier_synth_stats["F1"])))

	fig, ax = plt.subplots()
	ax.plot(classifier_stats["FP rate"], classifier_stats["TP rate"], label="Original")
	ax.plot(classifier_synth_stats["FP rate"], classifier_synth_stats["TP rate"], label="Synthetic")
	ax.legend()
	ax.set_xlabel("FP Rate")
	ax.set_ylabel("TP Rate")
	ax.set_title("ROC for Thyroid dataset")
	plt.show()

if __name__ == "__main__":
	app.run(main)








