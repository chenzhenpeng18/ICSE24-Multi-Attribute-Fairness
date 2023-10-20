import copy
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

def measure_final_score(dataset_orig_test, dataset_orig_predict, privileged_groups,unprivileged_groups, attr2):
    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    spd = abs(classified_metric_pred.statistical_parity_difference())
    aod = abs(classified_metric_pred.average_odds_difference())
    eod = abs(classified_metric_pred.equal_opportunity_difference())

    test_df = dataset_orig_test.convert_to_dataframe()[0]
    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['Probability'] = y_pred
    tt3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df, label_names=['Probability'],
                             protected_attribute_names=[attr2])
    tt4 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df_copy, label_names=['Probability'],
                             protected_attribute_names=[attr2])

    classified_metric_pred2 = ClassificationMetric(tt3, tt4, unprivileged_groups=[{attr2: 0}],
                                                   privileged_groups=[{attr2: 1}])

    spd2 = abs(classified_metric_pred2.statistical_parity_difference())
    aod2 = abs(classified_metric_pred2.average_odds_difference())
    eod2 = abs(classified_metric_pred2.equal_opportunity_difference())

    return accuracy, recall_macro, precision_macro, f1score_macro, mcc, spd, aod, eod, spd2, aod2, eod2
