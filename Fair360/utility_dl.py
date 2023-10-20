from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, MEPSDataset19, MEPSDataset21
import numpy as np
from tensorflow import keras

# def get_data(dataset_used, protected):
#     if dataset_used == "adult":
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'race': 1}]
#             unprivileged_groups = [{'race': 0}]
#         dataset_orig = AdultDataset().convert_to_dataframe()[0]
#         dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
#     elif dataset_used == "german":
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'age': 1}]
#             unprivileged_groups = [{'age': 0}]
#         dataset_orig = GermanDataset().convert_to_dataframe()[0]
#         dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
#         dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
#     elif dataset_used == "compas":
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'race': 1}]
#             unprivileged_groups = [{'race': 0}]
#         dataset_orig = CompasDataset().convert_to_dataframe()[0]
#         dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
#         dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
#     elif dataset_used == "mep1":
#         if protected == "RACE":
#             privileged_groups = [{'RACE': 1}]
#             unprivileged_groups = [{'RACE': 0}]
#         else:
#             privileged_groups = [{'SEX=2': 1}]
#             unprivileged_groups = [{'SEX=2': 0}]
#         dataset_orig = MEPSDataset19().convert_to_dataframe()[0].drop(columns=['SEX=1'])
#         dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
#     elif dataset_used == "mep2":
#         if protected == "RACE":
#             privileged_groups = [{'RACE': 1}]
#             unprivileged_groups = [{'RACE': 0}]
#         else:
#             privileged_groups = [{'SEX=2': 1}]
#             unprivileged_groups = [{'SEX=2': 0}]
#         dataset_orig = MEPSDataset21().convert_to_dataframe()[0].drop(columns=['SEX=1'])
#         dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
#     return dataset_orig, privileged_groups,unprivileged_groups

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
        clf.compile(loss="binary_crossentropy", optimizer="nadam",metrics=["accuracy"])
    return clf
