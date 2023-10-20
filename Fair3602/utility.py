from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, MEPSDataset19, MEPSDataset21
import numpy as np

# def get_data(dataset_used):
#     if dataset_used == "adult":
#         privileged_groups = [{'race': 1}, {'sex': 1}]
#         unprivileged_groups = [{'race': 0},{'sex': 0}]
#         dataset_orig = AdultDataset().convert_to_dataframe()[0]
#         dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
#     elif dataset_used == "german":
#         privileged_groups = [{'age': 1},{'sex': 1}]
#         unprivileged_groups = [{'age': 0},{'sex': 0}]
#         dataset_orig = GermanDataset().convert_to_dataframe()[0]
#         dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
#         dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
#     elif dataset_used == "compas":
#         privileged_groups = [{'race': 1},{'sex': 1}]
#         unprivileged_groups = [{'race': 0},{'sex': 0}]
#         dataset_orig = CompasDataset().convert_to_dataframe()[0]
#         dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
#         dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
#     elif dataset_used == "mep1":
#         privileged_groups = [{'RACE': 1}, {'SEX=2': 1}]
#         unprivileged_groups = [{'RACE': 0}, {'SEX=2': 0}]
#         dataset_orig = MEPSDataset19().convert_to_dataframe()[0].drop(columns=['SEX=1'])
#         dataset_orig.columns = dataset_orig.columns.str.replace("UTILIZATION", "Probability")
#     elif dataset_used == "mep2":
#         privileged_groups = [{'RACE': 1}, {'SEX=2': 1}]
#         unprivileged_groups = [{'RACE': 0}, {'SEX=2': 0}]
#         dataset_orig = MEPSDataset21().convert_to_dataframe()[0].drop(columns=['SEX=1'])
#         dataset_orig.columns = dataset_orig.columns.str.replace("UTILIZATION", "Probability")
#     return dataset_orig, privileged_groups,unprivileged_groups


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf
