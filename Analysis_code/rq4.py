import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean
from Fairea_multi.fairea import normalize,classify_region
from shapely.geometry import LineString

data = {}
for i in ['rf','lr','svm','dl']:
    data[i]={}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        data[i][j]={}
        for k in ['accuracy','precision','recall','f1score','mcc','spd1','aod1','eod1', 'spd2','aod2','eod2','spd','aod','eod']:
            data[i][j][k]={}

data_key_value_used = {1:'accuracy', 2: 'precision', 3: 'recall', 4: 'f1score', 5: 'mcc', 8: 'spd1', 11:'aod1', 14:'eod1', 17: 'spd2',20:'aod2',23:'eod2',28:'spd', 33:'aod',38:'eod'}
for j in ['lr','rf','svm','dl']:
    for name in ['default','rw','dir','eop','ceo','roc','fairsmote','maat','fairmask']:
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            fin = open('../Results_multi/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','pr','meta']:
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        fin = open('../Results_multi/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm', 'dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

fout = open('rq4_result','w')
fout.write('Results for Figure 2(a):\n')
fout.write('Results for datasets----------------------------\n')
for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    fout.write('\t'+dataset)
fout.write('\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        countt = {}
        countt['intersec'] = 0
        for j in ['lr', 'rf', 'svm', 'dl']:
            for i in ['spd','aod','eod']:
                if mean(data[j][dataset][i][name]) < mean(data[j][dataset][i]['default']):
                    countt['intersec']+=1
        fout.write('\t%f' % (countt['intersec']/12))
    fout.write('\n')

fout.write('\n')
fout.write('Results for models----------------------------\n')
for j in ['lr', 'rf', 'svm', 'dl']:
    fout.write('\t' + j)
fout.write('\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    for j in ['lr', 'rf', 'svm', 'dl']:
        countt = {}
        countt['intersec'] = 0
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            for i in ['spd','aod','eod']:
                if mean(data[j][dataset][i][name]) < mean(data[j][dataset][i]['default']):
                    countt['intersec']+=1
        fout.write('\t%f' % (countt['intersec']/15))
    fout.write('\n')

fout.write('\n')
fout.write('Results for metrics----------------------------\n')
for i in ['spd', 'aod', 'eod']:
    fout.write('\t'+i)
fout.write('\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    for i in ['spd', 'aod', 'eod']:
        countt = {}
        countt['intersec'] = 0
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            for j in ['lr', 'rf', 'svm', 'dl']:
                if mean(data[j][dataset][i][name]) < mean(data[j][dataset][i]['default']):
                    countt['intersec']+=1
        fout.write('\t%f' % (countt['intersec']/20))
    fout.write('\n')

fout.write('\n\n')

fout.write('Results for Figure 2(b):\n')

base_points = {}
for i in ['rf','lr','svm','dl']:
    base_points[i]={}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        base_points[i][j]={}

base_map = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 6: 'SPD', 7: 'AOD', 8:'EOD'}
for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    for i in ['rf', 'lr', 'svm','dl']:
        fin = open('../Fairea_baseline_multi/'+dataset+'_'+i+'_baseline','r')
        count = 0
        for line in fin:
            count += 1
            if count in base_map:
                base_points[i][dataset][base_map[count]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

data = {}
for i in ['rf','lr','svm','dl']:
    data[i]={}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD','AOD','EOD']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 28: 'SPD', 33: 'AOD', 38:'EOD'}
for j in ['lr','rf','svm','dl']:
    for name in ['default','rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            fin = open('../Results_multi/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','pr','meta']:
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        fin = open('../Results_multi/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm', 'dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

region_count = {}
for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['rf','lr','svm','dl']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in ['adv', 'meta', 'pr', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['rf','lr','svm','dl']:
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                for name in ['adv', 'meta', 'pr', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
                    methods = dict()
                    name_fair50 = data[i][j][fairmetric][name]
                    name_per50 = data[i][j][permetric][name]
                    for count in range(20):
                        methods[str(count)] = (float(name_per50[count]), float(name_fair50[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1

fout.write('Results for datasets----------------------------\n')
for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    fout.write('\t'+j)
fout.write('\n')
for name in ['adv', 'meta', 'pr', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
    fout.write(name)
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        final_count = {}
        for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
            final_count[region_kind] = 0
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
                for i in ['rf', 'lr', 'svm', 'dl']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
        final_sum = 0
        for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
            final_sum += final_count[region_kind]
        fout.write('\t%f' % ((final_count['good']+final_count['win-win'])/final_sum))
    fout.write('\n')

fout.write('\n')

fout.write('Results for models----------------------------\n')
for i in ['rf', 'lr', 'svm', 'dl']:
    fout.write('\t' + i)
fout.write('\n')
for name in ['adv', 'meta', 'pr', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
    fout.write(name)
    for i in ['rf', 'lr', 'svm', 'dl']:
        final_count = {}
        for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
            final_count[region_kind] = 0
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
                for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
        final_sum = 0
        for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
            final_sum += final_count[region_kind]
        fout.write('\t%f' % ((final_count['good']+final_count['win-win'])/final_sum))
    fout.write('\n')

fout.write('\n')

fout.write('Results for measurements----------------------------\n')
for fairmetric in ['SPD', 'AOD', 'EOD']:
    for permetric in ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']:
        fout.write('\t'+fairmetric+'-'+permetric)
fout.write('\n')
for name in ['adv', 'meta', 'pr', 'rw', 'dir', 'eop', 'ceo', 'roc', 'fairsmote', 'fairmask', 'maat']:
    fout.write(name)
    for fairmetric in ['SPD', 'AOD', 'EOD']:
        for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
            final_count = {}
            for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                final_count[region_kind] = 0
            for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
                for i in ['rf', 'lr', 'svm','dl']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
            final_sum = 0
            for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
                final_sum += final_count[region_kind]
            fout.write('\t%f' % ((final_count['good']+final_count['win-win'])/final_sum))
    fout.write('\n')

fout.close()
