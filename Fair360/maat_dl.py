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
import copy
from WAE import data_dis

from aif360.datasets import BinaryLabelDataset
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

dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race']}

multi_attr = macro_var[dataset_used]
for attr_tmp in multi_attr:
    if attr_tmp != attr:
        attr2 = attr_tmp

val_name = "maat_{}_{}_{}30.txt".format(clf_name,dataset_used,attr)
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
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    dataset_orig_train_new = data_dis(pd.DataFrame(dataset_orig_train),attr,dataset_used)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train_new)
    dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig.columns)
    dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                             protected_attribute_names=[attr])
    dataset_orig_train_new = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                            label_names=['Probability'],
                                            protected_attribute_names=macro_var[dataset_used])
    dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                             label_names=['Probability'],
                                             protected_attribute_names=macro_var[dataset_used])
    clf1 = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf1.fit(dataset_orig_train.features, dataset_orig_train.labels,epochs=20)

    clf2 = get_classifier(clf_name, dataset_orig_train_new.features.shape[1:])
    clf2.fit(dataset_orig_train_new.features, dataset_orig_train_new.labels,epochs=20)

    test_df_copy = copy.deepcopy(dataset_orig_test_1)
    pred_de1 = clf1.predict(dataset_orig_test_1.features).reshape(-1, 1)
    pred_de2 = clf2.predict(dataset_orig_test_2.features).reshape(-1, 1)

    res = []
    for i in range(len(pred_de1)):
        prob_t = (pred_de1[i]+pred_de2[i])/2
        if prob_t >= 0.5:
            res.append(1)
        else:
            res.append(0)

    test_df_copy.labels = np.array(res)

    round_result = measure_final_score(dataset_orig_test_1, test_df_copy, privileged_groups, unprivileged_groups, attr2)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()