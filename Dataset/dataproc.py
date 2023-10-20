import pandas as pd
import numpy as np

# dataset_orig = pd.read_csv("./Dataset/"+'adult' + "_processed.csv")
# p1 = len(dataset_orig[dataset_orig['Probability']==1])
# p0 = len(dataset_orig[dataset_orig['Probability']==0])
# print(p0/(p1+p0), p1/(p1+p0))

dataset_orig = pd.read_csv('adult.csv')
dataset_orig = dataset_orig.dropna()
dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)
dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)
dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 60 ) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 50 ) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 40 ) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 30 ) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 20 ) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 10 ) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])
dataset_orig.to_csv('adult_processed.csv',index=False)




dataset_orig = pd.read_csv('compas.csv')
dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)
dataset_orig = dataset_orig.dropna()
dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 0, 1)
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)
dataset_orig.to_csv('compas_processed.csv',index=False)




dataset_orig = pd.read_csv("default.csv")
dataset_orig = dataset_orig.dropna()
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0,1)
dataset_orig['AGE'] = np.where(dataset_orig['AGE'] >= 25, 0, 1)
dataset_orig = dataset_orig.drop(['ID'],axis=1)
dataset_orig = dataset_orig.rename(columns={"AGE" : "age"})
dataset_orig.to_csv('default_processed.csv',index=False)




MEPS15 = pd.read_csv('h181.csv')
MEPS15 = MEPS15.dropna()
MEPS15 = MEPS15.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})
MEPS15 = MEPS15[MEPS15['PANEL'] == 19]
MEPS15 = MEPS15[MEPS15['REGION'] >= 0] # remove values -1
MEPS15 = MEPS15[MEPS15['AGE'] >= 0] # remove values -1
MEPS15 = MEPS15[MEPS15['MARRY'] >= 0] # remove values -1, -7, -8, -9
MEPS15 = MEPS15[MEPS15['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
MEPS15 = MEPS15[(MEPS15[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]
# ## Change symbolics to numerics
MEPS15['RACEV2X'] = np.where((MEPS15['HISPANX'] == 2) & (MEPS15['RACEV2X'] == 1), 1, MEPS15['RACEV2X'])
MEPS15['RACEV2X'] = np.where(MEPS15['RACEV2X'] != 1 , 0, MEPS15['RACEV2X'])
MEPS15 = MEPS15.rename(columns={"RACEV2X" : "RACE"})
# MEPS15['UTILIZATION'] = np.where(MEPS15['UTILIZATION'] >= 10, 1, 0)
def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']
MEPS15['TOTEXP15'] = MEPS15.apply(lambda row: utilization(row), axis=1)
lessE = MEPS15['TOTEXP15'] < 10.0
MEPS15.loc[lessE,'TOTEXP15'] = 0.0
moreE = MEPS15['TOTEXP15'] >= 10.0
MEPS15.loc[moreE,'TOTEXP15'] = 1.0
MEPS15 = MEPS15.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
MEPS15 = MEPS15[['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                 'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']]
MEPS15['SEX'] = np.where(MEPS15['SEX'] == 2, 1, 0)
MEPS15 = MEPS15.rename(columns={"UTILIZATION": "Probability","RACE" : "race", "SEX": "sex"})
MEPS15.to_csv('mep1_processed.csv',index=False)





MEPS16 = pd.read_csv('h192.csv')
MEPS16 = MEPS16.dropna()
MEPS16 = MEPS16.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT16' : 'POVCAT', 'INSCOV16' : 'INSCOV'})
MEPS16 = MEPS16[MEPS16['PANEL'] == 21]
MEPS16 = MEPS16[MEPS16['REGION'] >= 0] # remove values -1
MEPS16 = MEPS16[MEPS16['AGE'] >= 0] # remove values -1
MEPS16 = MEPS16[MEPS16['MARRY'] >= 0] # remove values -1, -7, -8, -9
MEPS16 = MEPS16[MEPS16['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
MEPS16 = MEPS16[(MEPS16[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]
# ## Change symbolics to numerics
MEPS16['RACEV2X'] = np.where((MEPS16['HISPANX'] == 2) & (MEPS16['RACEV2X'] == 1), 1, MEPS16['RACEV2X'])
MEPS16['RACEV2X'] = np.where(MEPS16['RACEV2X'] != 1 , 0, MEPS16['RACEV2X'])
MEPS16 = MEPS16.rename(columns={"RACEV2X" : "RACE"})
def utilization(row):
        return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']
MEPS16['TOTEXP16'] = MEPS16.apply(lambda row: utilization(row), axis=1)
lessE = MEPS16['TOTEXP16'] < 10.0
MEPS16.loc[lessE,'TOTEXP16'] = 0.0
moreE = MEPS16['TOTEXP16'] >= 10.0
MEPS16.loc[moreE,'TOTEXP16'] = 1.0
MEPS16 = MEPS16.rename(columns = {'TOTEXP16' : 'UTILIZATION'})
MEPS16 = MEPS16[['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                 'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT16F']]
MEPS16['SEX'] = np.where(MEPS16['SEX'] == 2, 1, 0)
MEPS16 = MEPS16.rename(columns={"UTILIZATION": "Probability","RACE" : "race", "SEX": "sex"})
MEPS16.to_csv('mep2_processed.csv',index=False)