from numpy import mean
import scipy.stats as stats
import pandas as pd
from scipy.stats import spearmanr
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]


metric_list = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2', 'eod2']
data = {}
for i in ['rf', 'lr', 'svm', 'dl']:
    data[i] = {}
    for j in ['adult-sex', 'adult-race', 'compas-sex', 'compas-race', 'default-sex', 'default-age', 'mep1-sex', 'mep1-race', 'mep2-sex', 'mep2-race']:
        data[i][j] = {}
        for k in metric_list:
            data[i][j][k] = {}

for j in ['rf', 'lr', 'svm', 'dl']:
    for name in ['default', 'rw', 'dir', 'eop', 'roc', 'ceo','fairsmote', 'fairmask', 'maat']:
        for dataset in ['adult-sex', 'adult-race', 'compas-sex', 'compas-race', 'default-sex', 'default-age', 'mep1-sex', 'mep1-race', 'mep2-sex', 'mep2-race']:
            (dataset_pre, dataset_aft) = dataset.split('-')
            fin = open('../Results/' + name + '_' + j + '_' + dataset_pre + '_' + dataset_aft + '.txt', 'r')
            count = 0
            for line in fin:
                data[j][dataset][metric_list[count]][name] = list(map(float, line.strip().split('\t')[1:21]))
                count = count + 1
            fin.close()

for name in ['adv', 'meta', 'pr']:
    for dataset in ['adult-sex', 'adult-race', 'compas-sex', 'compas-race', 'default-sex', 'default-age', 'mep1-sex', 'mep1-race', 'mep2-sex', 'mep2-race']:
        (dataset_pre, dataset_aft) = dataset.split('-')
        fin = open('../Results/' + name + '_lr_' + dataset_pre + '_' + dataset_aft + '.txt', 'r')
        count = 0
        for line in fin:
            for j in ['rf', 'lr', 'svm', 'dl']:
                data[j][dataset][metric_list[count]][name] = list(map(float, line.strip().split('\t')[1:21]))
            count = count + 1
        fin.close()

file_name = "rq1_result"
fout = open(file_name, 'w')
fout.write("Results for Table 5:\n")
fout.write('\tfairness_decrease\tfairness_decrease_large_effect\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    countt = {}
    countt['decrease'] = 0
    countt['decrease_large_effect'] =0
    for z in ['adult-sex', 'adult-race', 'compas-sex', 'compas-race', 'default-sex', 'default-age', 'mep1-sex', 'mep1-race', 'mep2-sex', 'mep2-race']:
        for clf in ['lr','rf','svm','dl']:
            for k in ['spd2', 'aod2', 'eod2']:
                default_list = data[clf][z][k]['default']
                real_list = data[clf][z][k][name]
                default_valuefork = mean(default_list)
                real_valuefork = mean(real_list)
                if default_valuefork < real_valuefork:
                    countt['decrease']+=1
                    if mann(default_list, real_list) < 0.05 and abs(cliffs_delta(default_list, real_list)[0]) >=0.428:
                        countt['decrease_large_effect']+=1
    fout.write('\t%f\t%f\n' % (countt['decrease']/120,countt['decrease_large_effect']/120))

fout.write('\n')

fout.write("Results for Table 6:\n")

for dataset in ['adult','compas','default','mep1','mep2']:
    fout.write(dataset+':\n')
    pa1 = 'sex'
    pa2 = 'race'
    if dataset == 'default':
        pa2 = 'age'
    dataset_orig = pd.read_csv("../Dataset/"+dataset+ "_processed.csv")
    fout.write(str(spearmanr(dataset_orig[pa1],dataset_orig[pa2])))
    fout.write('\n')

fout.write('\n')

fout.write("Correlation between the last two columns:\n")
fout.write(str(spearmanr([0.101, 0.101, 0.068, 0.068, -0.069, -0.069, -0.015, -0.015, -0.016, -0.016],[0.447, 0.568, 0.409, 0.333, 0.841, 0.765, 0.742, 0.644, 0.485, 0.515])))

fout.close()
