from numpy import mean, std, sqrt
import scipy.stats as stats


def cohen_d(x, y):
    return abs(mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)


def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

def meann(x):
    if len(x) ==0:
        return 0
    else:
        return mean(x)

def maxx(x):
    if len(x) ==0:
        return 0
    else:
        return max(x)


metric_list = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2', 'eod2']
data = {}
for i in ['rf', 'lr', 'svm', 'dl']:
    data[i] = {}
    for j in ['adult-sex', 'adult-race', 'compas-sex', 'compas-race', 'default-sex', 'default-age', 'mep1-sex', 'mep1-race', 'mep2-sex', 'mep2-race']:
        data[i][j] = {}
        for k in metric_list:
            data[i][j][k] = {}

for j in ['rf', 'lr', 'svm', 'dl']:
    for name in ['default', 'rw', 'dir', 'eop', 'ceo',  'roc', 'fairsmote', 'fairmask', 'maat']:
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

metric_multi = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod']
data_multi = {}
for i in ['rf', 'lr', 'svm', 'dl']:
    data_multi[i] = {}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        data_multi[i][j] = {}
        for k in metric_multi:
            data_multi[i][j][k] = {}

data_key_value_used = {1:'accuracy', 2: 'precision', 3: 'recall', 4: 'f1score', 5: 'mcc', 28: 'spd', 33: 'aod', 38:'eod'}
for j in ['lr','rf','svm','dl']:
    for name in ['default', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            fin = open('../Results_multi/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data_multi[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','pr','meta']:
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        fin = open('../Results_multi/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm', 'dl']:
                    data_multi[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()


fout = open('rq3_1_result','w')
fout.write("Results for Table 9:\n")
print_list = ['single_abso', 'single_rela', 'multi_abso', 'multi_rela']
for k in ['accuracy', 'precision','recall', 'f1score','mcc']:
    for pp in print_list:
        fout.write('\t'+k+"_"+pp)
fout.write('\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    for k in ['accuracy', 'precision','recall', 'f1score','mcc']:
        compto_default_single = []
        compto_default_multi = []
        compto_default_single_rela = []
        compto_default_multi_rela = []
        for z in ['adult-sex', 'adult-race', 'compas-sex','compas-race', 'default-sex','default-age', 'mep1-sex','mep1-race', 'mep2-sex','mep2-race']:
            for clf in ['lr','rf','svm','dl']:
                (dataset_pre, dataset_aft) = z.split('-')
                a1_perf = mean(data[clf][z][k][name])
                a2_perf = mean(data_multi[clf][dataset_pre][k][name])
                compto_default_single.append(a1_perf-mean(data[clf][z][k]['default']))
                compto_default_single_rela.append((a1_perf-mean(data[clf][z][k]['default']))/mean(data[clf][z][k]['default']))
                compto_default_multi.append(a2_perf-mean(data[clf][z][k]['default']))
                compto_default_multi_rela.append((a2_perf-mean(data[clf][z][k]['default']))/mean(data[clf][z][k]['default']))
        fout.write('\t%f\t%f\t%f\t%f' % (mean(compto_default_single), mean(compto_default_multi),mean(compto_default_single_rela),mean(compto_default_multi_rela)))
    fout.write('\n')
fout.close()

