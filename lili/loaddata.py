import pandas as pd 

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import os

os.chdir('/home/jlm217/rds/rds-mrc-bsu-csoP2nj6Y6Y/mimah/stopah/stopah/data') 

pd.set_option('display.max_columns', None)  

#Open TSV file 
#df = pd.read_csv('data_clinical.tsv',sep='\t')
#df.columns = df.columns.str.removesuffix('\\')
#df.columns = df.columns.str.removeprefix('\\Private Studies\\STOPAH1\\')
#Open R files 

ro.r['load']('STOPAH_ForSolon.RData')

#Function that outputs all R files 

#There are 3 data sets: data, stopah and stopah.day7

def R_dataset(x):
    with (ro.default_converter + pandas2ri.converter).context():
        stopah = ro.conversion.get_conversion().rpy2py(ro.r[x])
    return stopah

stopah = R_dataset('stopah')

stopah.to_csv('/home/jlm217/rds/rds-mrc-bsu-csoP2nj6Y6Y/mimah/stopah/stopah/data/stopah.csv')

selected = ['D28_DTH','D90_DTH','Prednisolone']

baseline = ['Gender','Baseline_sepsis','Baseline_GIB',
'Age.at.randomisation..calc.','Weight','Max.grams.of.alcohol.drunk.per.day..calc.',
'Hepatic.Encephalopathy...Merged',
'Temperature...Merged',
'Pulse...Merged',
'Systolic.BP...Merged',
'Diastolic.BP...Merged',
'MAP',
'Hb...Merged',
'Platelets...Merged',
'WBC...Merged',
'Neutrophils...Merged',
'INR...Merged.clinical.and.calc',
'Bilirubin.Merged',
'ALT...Merged',
'ALP...Merged',
'Albumin...Merged',
'Sodium...Merged',
'Potassium...Merged',
'Urea...Merged',
'Creatinine...Merged',
'NLR_0',
'bDNA',
'Ferritin_ngml',
'Iron_mumoll',
'Transferrin',
'TSAT',
'PNPLA3_Add',
'PNPLA3_Rec',
'HPCT_NG'] 

#reduce data set to target, baselines and treatment

stopah = stopah[selected+baseline]

stopah.reset_index(drop=True, inplace=True)



