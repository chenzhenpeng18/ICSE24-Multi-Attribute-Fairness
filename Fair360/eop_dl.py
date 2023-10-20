import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility_dl import get_classifier
from sklearn.model_selection import train_test_split
import argparse

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'default', 'compas', 'mep1', 'mep2'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race']}

multi_attr = macro_var[dataset_used]
for attr_tmp in multi_attr:
    if attr_tmp != attr:
        attr2 = attr_tmp

randseed = 12345679

val_name = "eop_{}_{}_{}30.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
privileged_groups = [{attr: 1}]
unprivileged_groups = [{attr: 0}]
privileged_groups2 = [{attr2: 1}]
unprivileged_groups2 = [{attr2: 0}]

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2', 'eod2']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 30
for r in range(20,50):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=[attr,attr2])

    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)

    train_pred = clf.predict_classes(dataset_orig_train.features).reshape(-1, 1)
    train_prob = clf.predict(dataset_orig_train.features).reshape(-1, 1)

    pred = clf.predict_classes(dataset_orig_test.features).reshape(-1, 1)
    pred_prob = clf.predict(dataset_orig_test.features).reshape(-1, 1)

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = train_pred
    dataset_orig_train_pred.scores = train_prob

    dataset_orig_test_pred = dataset_orig_test.copy()
    dataset_orig_test_pred.labels = pred
    dataset_orig_test_pred.scores = pred_prob
    
    eqo = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     seed=randseed)
    eqo = eqo.fit(dataset_orig_train, dataset_orig_train_pred)
    pred_eqo = eqo.predict(dataset_orig_test_pred)

    round_result = measure_final_score(dataset_orig_test, pred_eqo, privileged_groups, unprivileged_groups, attr2)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()