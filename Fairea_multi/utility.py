from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from aif360.datasets import BinaryLabelDataset,AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19,MEPSDataset21,LawSchoolGPADataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
import pandas as pd

def get_data(dataset_used, protected):
    if dataset_used == "adult" or dataset_used == "mep1" or dataset_used == "mep2":
        mutation_strategy  = {"0":[1,0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()
        dataset_orig = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig,
                                            label_names=['Probability'],
                                            protected_attribute_names=['sex','race'])
    elif dataset_used == "default":
        mutation_strategy = {"0": [1, 0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()
        dataset_orig = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig,
                                          label_names=['Probability'],
                                          protected_attribute_names=['sex', 'age'])
    elif dataset_used == "compas":
        mutation_strategy = {"1": [0, 1]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()
        dataset_orig = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig,
                                          label_names=['Probability'],
                                          protected_attribute_names=['sex', 'race'])
    return dataset_orig, privileged_groups,unprivileged_groups,mutation_strategy


# def get_data(dataset_used, protected):
#     """ Obtains dataset from AIF360.
#
#     Parameters:
#         dataset_used (str) -- Name of the dataset
#         protected (str)    -- Protected attribute used
#     Returns:
#         dataset_orig (dataset)     -- Classifier with default configuration from scipy
#         dataset_orig (dataset)     -- Classifier with default configuration from scipy
#         privileged_groups (list)   -- Attribute and corresponding value of privileged group
#         unprivileged_groups (list) -- Attribute and corresponding value of unprivileged group
#         optim_options (dict)       -- Options if provided by AIF360
#     """
#     if dataset_used == "adult":
#         mutation_strategy  = {"0":[1,0]}
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'race': 1}]
#             unprivileged_groups = [{'race': 0}]
#         dataset_orig = AdultDataset()
#     elif dataset_used == "german":
#         mutation_strategy = {"1": [0, 1]}
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'age': 1}]
#             unprivileged_groups = [{'age': 0}]
#         dataset_orig = GermanDataset()
#         label_list = dataset_orig.labels
#         label_list[label_list == 2] = 0
#         dataset_orig.labels = label_list
#         dataset_orig.unfavorable_label = 0
#     elif dataset_used == "compas":
#         mutation_strategy = {"0": [1, 0]}
#         if protected == "sex":
#             privileged_groups = [{'sex': 1}]
#             unprivileged_groups = [{'sex': 0}]
#         else:
#             privileged_groups = [{'race': 1}]
#             unprivileged_groups = [{'race': 0}]
#         dataset_orig = CompasDataset()
#     elif dataset_used == "mep1":
#         mutation_strategy = {"0": [1, 0]}
#         if protected == 'RACE':
#             privileged_groups = [{'RACE': 1}]
#             unprivileged_groups = [{'RACE': 0}]
#         else:
#             privileged_groups = [{'SEX=2': 1}]
#             unprivileged_groups = [{'SEX=2': 0}]
#         dataset_orig = MEPSDataset19().convert_to_dataframe()[0].drop(columns = ['SEX=1'])
#         dataset_orig = BinaryLabelDataset(df=dataset_orig, label_names=['UTILIZATION'], protected_attribute_names = ['RACE','SEX=2'], favorable_label=1, unfavorable_label=0)
#     elif dataset_used == "mep2":
#         mutation_strategy = {"0": [1, 0]}
#         if protected == 'RACE':
#             privileged_groups = [{'RACE': 1}]
#             unprivileged_groups = [{'RACE': 0}]
#         else:
#             privileged_groups = [{'SEX=2': 1}]
#             unprivileged_groups = [{'SEX=2': 0}]
#         dataset_orig = MEPSDataset21().convert_to_dataframe()[0].drop(columns = ['SEX=1'])
#         dataset_orig = BinaryLabelDataset(df=dataset_orig, label_names=['UTILIZATION'], protected_attribute_names = ['RACE','SEX=2'], favorable_label=1, unfavorable_label=0)
#
#     return dataset_orig, privileged_groups,unprivileged_groups,mutation_strategy