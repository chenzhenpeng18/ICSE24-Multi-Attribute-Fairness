import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath('..'))
from Measure_new import measure_final_score
from Generate_Samples import generate_samples
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

def situation(clf, X_train,y_train,keyword,keyword2):
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    same = [1 if each else 0 for each in same]

    X_flip2 = X_train.copy()
    X_flip2[keyword2] = np.where(X_flip[keyword2] == 1, 0, 1)
    c = np.array(clf.predict(X_flip2))
    same2 = (a==c)
    same2 = [1 if each else 0 for each in same2]

    X_train['same'] = same
    X_train['same2'] = same2
    X_train['y'] = y_train
    X_rest = X_train[(X_train['same']==1)&(X_train['same2']==1)]
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','same2','y'])
    return X_rest,y_rest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'default', 'compas', 'mep1', 'mep2'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['rf', 'svm', 'lr'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'],'default':['sex','age'], 'mep1': ['sex','race'], 'mep2': ['sex','race']}

    protected_attribute1 = macro_var[dataset_used][0]
    protected_attribute2 = macro_var[dataset_used][1]

    val_name = "fairsmote_{}_{}_multi.txt".format(clf_name, dataset_used)
    fout = open(val_name, 'w')

    results = {}
    performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'wcspd-00','wcspd-01','wcspd-10','wcspd-11','wcspd', 'wcaod-00','wcaod-01','wcaod-10','wcaod-11','wcaod', 'wceod-00','wceod-01','wceod-10','wceod-11','wceod']
    for p_index in performance_index:
        results[p_index] = []
    
    dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
    
    repeat_time = 20
    for i in range(repeat_time):
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
                    dataset_orig_train[protected_attribute1] == 0)
                                                & (dataset_orig_train[protected_attribute2] == 0)])
        zero_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (
                    dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 1)])
        zero_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (
                    dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        zero_one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (
                    dataset_orig_train[protected_attribute1] == 1)
                                              & (dataset_orig_train[protected_attribute2] == 1)])
        one_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (
                    dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        one_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (
                    dataset_orig_train[protected_attribute1] == 0)
                                              & (dataset_orig_train[protected_attribute2] == 1)])
        one_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (
                    dataset_orig_train[protected_attribute1] == 1)
                                              & (dataset_orig_train[protected_attribute2] == 0)])
        one_one_one = len(dataset_orig_train[
                              (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                              & (dataset_orig_train[protected_attribute2] == 1)])

        maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)
        zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
        zero_zero_one_to_be_incresed = maximum - zero_zero_one
        zero_one_zero_to_be_incresed = maximum - zero_one_zero
        zero_one_one_to_be_incresed = maximum - zero_one_one
        one_zero_zero_to_be_incresed = maximum - one_zero_zero
        one_zero_one_to_be_incresed = maximum - one_zero_one
        one_one_zero_to_be_incresed = maximum - one_one_zero
        one_one_one_to_be_incresed = maximum - one_one_one

        df_zero_zero_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
            & (dataset_orig_train[protected_attribute2] == 0)]
        df_zero_zero_one = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
            & (dataset_orig_train[protected_attribute2] == 1)]
        df_zero_one_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
            & (dataset_orig_train[protected_attribute2] == 0)]
        df_zero_one_one = dataset_orig_train[
            (dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
            & (dataset_orig_train[protected_attribute2] == 1)]
        df_one_zero_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
            & (dataset_orig_train[protected_attribute2] == 0)]
        df_one_zero_one = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
            & (dataset_orig_train[protected_attribute2] == 1)]
        df_one_one_zero = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
            & (dataset_orig_train[protected_attribute2] == 0)]
        df_one_one_one = dataset_orig_train[
            (dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
            & (dataset_orig_train[protected_attribute2] == 1)]

        for cate in [protected_attribute1, protected_attribute2]:
            df_zero_zero_zero[cate] = df_zero_zero_zero[cate].astype(str)
            df_zero_zero_one[cate] = df_zero_zero_one[cate].astype(str)
            df_zero_one_zero[cate] = df_zero_one_zero[cate].astype(str)
            df_zero_one_one[cate] = df_zero_one_one[cate].astype(str)
            df_one_zero_zero[cate] = df_one_zero_zero[cate].astype(str)
            df_one_zero_one[cate] = df_one_zero_one[cate].astype(str)
            df_one_one_zero[cate] = df_one_one_zero[cate].astype(str)
            df_one_one_one[cate] = df_one_one_one[cate].astype(str)

        print("Start generating samples...")
        df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed, df_zero_zero_zero,'')
        df_zero_zero_one = generate_samples(zero_zero_one_to_be_incresed, df_zero_zero_one,'')
        df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed, df_zero_one_zero,'')
        df_zero_one_one = generate_samples(zero_one_one_to_be_incresed, df_zero_one_one,'')
        df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed, df_one_zero_zero,'')
        df_one_zero_one = generate_samples(one_zero_one_to_be_incresed, df_one_zero_one,'')
        df_one_one_zero = generate_samples(one_one_zero_to_be_incresed, df_one_one_zero,'')
        df_one_one_one = generate_samples(one_one_one_to_be_incresed, df_one_one_one,'')
        df = pd.concat([df_zero_zero_zero, df_zero_zero_one, df_zero_one_zero, df_zero_one_one,
                        df_one_zero_zero, df_one_zero_one, df_one_one_zero, df_one_one_one])

        df.columns = dataset_orig.columns
        clf2 = RandomForestClassifier()
        clf2.fit(X_train, y_train)
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        print("Situational testing...")
        X_train, y_train = situation(clf2, X_train, y_train, protected_attribute1, protected_attribute2)

        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
            'Probability']

        clf = get_classifier(clf_name)
        round_result = measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test,
                                           protected_attribute1, protected_attribute2)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()