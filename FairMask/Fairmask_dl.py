import pandas as pd
import numpy as np
import copy
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from Measure_new import measure_final_score
import argparse
import tensorflow as tf
from tensorflow import keras

def get_classifier(name,datasize):
    if name == "dl":
        clf = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=datasize),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    return clf

def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'default', 'compas', 'mep1', 'mep2'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['dl'], help="Classifier name")
    parser.add_argument("-p", "--protected", type=str, required=True,
                        help="Protected attribute")

    args = parser.parse_args()
    dataset_used = args.dataset
    attr = args.protected
    clf_name = args.clf

    macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'],'default':['sex','age'], 'mep1': ['sex','race'], 'mep2': ['sex','race']}

    multi_attr = macro_var[dataset_used]

    for attr_tmp in multi_attr:
        if attr_tmp != attr:
            attr2 = attr_tmp

    val_name = "fairmask_{}_{}_{}30.txt".format(clf_name, dataset_used, attr)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2',
                         'eod2']
    for p_index in performance_index:
        results[p_index] = []

    dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()

    repeat_time = 30
    for i in range(20,50):
        print(i)
        np.random.seed(i)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        reduced = list(X_train.columns)
        reduced.remove(attr)
        X_reduced, y_reduced = X_train.loc[:, reduced], X_train[attr]
        # Build model to predict the protect attribute
        clf1 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
        clf = get_classifier(clf_name, (X_trains.shape[1],))
        clf.fit(X_trains, y_trains,epochs=20)
        y_proba = clf.predict(X_trains)
        if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
            clf1.fit(X_trains, y_trains)
        else:
            clf1.fit(X_trains, y_proba)

        X_test_reduced = X_test.loc[:, X_test.columns != attr]
        protected_pred = clf1.predict(X_test_reduced)
        if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
            protected_pred = reg2clf(protected_pred, threshold=0.5)

        # Build model to predict the target attribute Y
        clf2 = get_classifier(clf_name,(X_train.shape[1],))
        clf2.fit(X_train, y_train, epochs=20)
        X_test.loc[:, attr] = protected_pred
        y_pred = clf2.predict_classes(X_test)

        round_result = measure_final_score(dataset_orig_test, y_test, y_pred, attr, attr2)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()
