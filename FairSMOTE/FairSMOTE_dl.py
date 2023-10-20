import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath('..'))
from Measure_dl import measure_final_score
from Generate_Samples import generate_samples
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

def situation(clf, X_train,y_train,keyword):
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    same = [1 if each else 0 for each in same]
    X_train['same'] = same
    X_train['y'] = y_train
    X_rest = X_train[X_train['same']==1]
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','y'])
    return X_rest,y_rest

# dataset_orig = pd.read_csv("./Data/"+'german' + "_processed.csv").dropna()
# print(len(dataset_orig[dataset_orig['Probability']==0]))
# print(len(dataset_orig[dataset_orig['Probability']==1]))
# print(len(dataset_orig[(dataset_orig['Probability']==0)& (
#                 dataset_orig['sex'] == 0)]))
# print(len(dataset_orig[(dataset_orig['Probability']==1)& (
#                 dataset_orig['sex'] == 0)]))
# print(len(dataset_orig[(dataset_orig['Probability']==0)& (
#                 dataset_orig['sex'] == 1)]))
# print(len(dataset_orig[(dataset_orig['Probability']==1)& (
#                 dataset_orig['sex'] == 1)]))
# print(len(dataset_orig[(dataset_orig['Probability']==0)& (
#                 dataset_orig['age'] == 0)]))
# print(len(dataset_orig[(dataset_orig['Probability']==1)& (
#                 dataset_orig['age'] == 0)]))
# print(len(dataset_orig[(dataset_orig['Probability']==0)& (
#                 dataset_orig['age'] == 1)]))
# print(len(dataset_orig[(dataset_orig['Probability']==1)& (
#                 dataset_orig['age'] == 1)]))


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

    val_name = "fairsmote_{}_{}_{}30.txt".format(clf_name, dataset_used, attr)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2',
                         'eod2']
    for p_index in performance_index:
        results[p_index] = []
    
    dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
    
    repeat_time = 30
    for i in range(20,50):
        print(i)
        np.random.seed(i)
        
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle = True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train[
            'Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
            'Probability']

        zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (
                dataset_orig_train[attr] == 0)])

        zero_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (
                dataset_orig_train[attr] == 1)])

        one_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (
                dataset_orig_train[attr] == 0)])

        one_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (
                dataset_orig_train[attr] == 1)])

        maximum = max(zero_zero_zero, zero_one_zero, one_zero_zero, one_one_zero)
        zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
        zero_one_zero_to_be_incresed = maximum - zero_one_zero
        one_zero_zero_to_be_incresed = maximum - one_zero_zero
        one_one_zero_to_be_incresed = maximum - one_one_zero
        df_zero_zero_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 0)]
        df_zero_one_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[attr] == 1)]
        df_one_zero_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 0)]
        df_one_one_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 1)]

        df_zero_zero_zero[attr] = df_zero_zero_zero[attr].astype(str)
        df_zero_one_zero[attr] = df_zero_one_zero[attr].astype(str)
        df_one_zero_zero[attr] = df_one_zero_zero[attr].astype(str)
        df_one_one_zero[attr] = df_one_one_zero[attr].astype(str)

        print("Start generating samples...")
        df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed, df_zero_zero_zero, '')
        df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed, df_zero_one_zero, '')
        df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed, df_one_zero_zero, '')
        df_one_one_zero = generate_samples(one_one_zero_to_be_incresed, df_one_one_zero, '')
        df = pd.concat([df_zero_zero_zero, df_zero_one_zero, df_one_zero_zero, df_one_one_zero])

        df.columns = dataset_orig.columns
        clf2 = RandomForestClassifier()
        clf2.fit(X_train, y_train)
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        print("Situational testing...")
        X_train, y_train = situation(clf2, X_train, y_train, attr)
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
            'Probability']

        clf = get_classifier(clf_name,(X_train.shape[1],))
        round_result = measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test,
                                           attr, attr2)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()