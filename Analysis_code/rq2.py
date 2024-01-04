import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

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


fout = open('rq2_result','w')
fout.write("Results for Table 7:\n")
fout.write('\tintersec_increase\tintersec_increase_large_effect\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    countt = {}
    countt['intersec'] = 0
    countt['intersec_sig'] = 0
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        for j in ['lr', 'rf', 'svm', 'dl']:
            for i in ['spd','aod','eod']:
                if mean(data[j][dataset][i][name]) < mean(data[j][dataset][i]['default']):
                    countt['intersec']+=1
                    if mann(data[j][dataset][i][name], data[j][dataset][i]['default']) < 0.05 and abs(cliffs_delta(data[j][dataset][i][name], data[j][dataset][i]['default'])[0]) >=0.428:
                        countt['intersec_sig']+=1
    fout.write('\t%f\t%f\n' % (countt['intersec']/60,countt['intersec_sig']/60))

fout.write('\n')
fout.write("Results for Table 8:\n")

fout.write('\tSPD\tAOD\tEOD\n')
for name in ['rw', 'dir', 'meta','adv', 'pr', 'eop', 'ceo','roc', 'fairsmote', 'maat', 'fairmask']:
    fout.write(name)
    value_list = {}
    relative_list = {}
    for i in ['spd', 'aod', 'eod']:
        value_list[i] = []
        relative_list[i]=[]
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            for j in ['lr', 'rf', 'svm', 'dl']:
                value_list[i].append(mean(data[j][dataset][i][name])-mean(data[j][dataset][i]['default']))
                relative_list[i].append(100*(mean(data[j][dataset][i][name])-mean(data[j][dataset][i]['default']))/mean(data[j][dataset][i]['default']))
        fout.write('\t%.3f (%.1f' % (mean(value_list[i]),mean(relative_list[i])))
        fout.write('\\%)')
    fout.write('\n')
fout.close()
