import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_classifier
from sklearn.model_selection import train_test_split
import os
import argparse

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'default', 'compas', 'mep1', 'mep2'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")
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


val_name = "dir_{}_{}_{}30.txt".format(clf_name,dataset_used,attr)
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

    di = DisparateImpactRemover(sensitive_attribute=attr)
    train_repd = di.fit_transform(dataset_orig_train)
    test_repd = di.fit_transform(dataset_orig_test)
    index = dataset_orig_train.feature_names.index(attr)
    X_tr = np.delete(train_repd.features, index, axis=1)
    X_te = np.delete(test_repd.features, index, axis=1)
    y_tr = train_repd.labels.ravel()

    clf = get_classifier(clf_name)
    clf.fit(X_tr, y_tr)

    test_repd_pred = test_repd.copy()
    test_repd_pred.labels = clf.predict(X_te)

    round_result = measure_final_score(test_repd, test_repd_pred, privileged_groups, unprivileged_groups,attr2)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()