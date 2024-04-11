import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('stopah.csv')

for x in df.columns:
        if len(df[df[x] == -2147483648].index) >0:
            ix = df[df[x] == -2147483648].index
            imp = df[[x]].loc[~df.index.isin(ix)].mean()[0]
            df.loc[ix,x] = np.nan

# Function that returns subset of interest 

def dataset(df,subset,treatment,missing,split,size):
    #-df: Pandas data fame
    #-subset: String, determines subset 'Lille','7day','Baseline','Paper'
    #-treatment: String, 'Yes' and 'No'
    #-split: String, determines split 'train-test','mixed'
    #-testsize: Float, Size of test set 
    #
    #Subsets 
    #
    # --- Baseline ---
    #
    #Make all funny values NA
    
    
    baseline = ['Gender','Baseline_sepsis','Baseline_GIB','Age.at.randomisation..calc.','Weight','Max.grams.of.alcohol.drunk.per.day..calc.','Hepatic.Encephalopathy...Merged','Temperature...Merged','Pulse...Merged','Systolic.BP...Merged','Diastolic.BP...Merged','MAP','Hb...Merged','Platelets...Merged','WBC...Merged','Neutrophils...Merged','INR...Merged.clinical.and.calc','Bilirubin.Merged','ALT...Merged','ALP...Merged','Albumin...Merged','Sodium...Merged','Potassium...Merged','Urea...Merged','Creatinine...Merged','NLR_0','bDNA','Ferritin_ngml','Iron_mumoll','Transferrin','TSAT','PNPLA3_Add','PNPLA3_Rec','HPCT_NG'] 
    #
    # --- Seven day ---
    #
    sevenday = ['Hepatic.Encephalopathy...Treatment.Day.7..','Day.7.infection',
    'Gastrointestinal.Bleed.since.the.last.visit.Gastrointestinal.bleed...and.Choose..Treatment.Day.7..',
    'Temperature..Treatment.Day.7..','Pulse..Treatment.Day.7..',
    'Systolic.BP..Treatment.Day.7..','Diastolic.BP..Treatment.Day.7..','MAP..Treatment.Day.7',
    'Hb..1.decimal.point..Haematology..Treatment.Day.7..','Platelets.day.7','WBC.day.7',
    'Neutrophils.day.7','INR.clinical.and.calc.day.7','Bilirubin.day.7','ALT.day.7',
    'ALP.day.7','Albumin.day.7','Sodium.day.7','Potassium.day.7','Urea.day.7','Creatinine.day.7']
    #
    #Lille 
    #
    Lille_var = ['Age.at.randomisation..calc.','Albumin...Merged','Creatinine..mg.dL....Merged',
             'Bilirubin.Merged','Prothrombin.Time..patient....Merged','Bilirubin.day.7']
    ### Subset 
    if subset == 'Lille':
        df = df[Lille_var+['Prednisolone','D90_DTH']]
        
        df['Renal Insufficency'] = (df['Creatinine..mg.dL....Merged'].loc[:] > 1.3).astype('float')
        df = df.drop(['Creatinine..mg.dL....Merged'],axis=1)
        #df['Bilirubin.day.7'] = df['Bilirubin.day.7'] - df['Bilirubin.Merged']
         
    elif subset == '7day':
        df = df[sevenday+['Prednisolone','D90_DTH']]
    elif subset == 'baseline':
        df = df[baseline+['Prednisolone','D90_DTH']]
    elif subset == 'full':
        df = df[sevenday+baseline+['Prednisolone','D90_DTH']]
    
    #Treatment 
    if treatment == 'yes':
        df = df[df['Prednisolone']==1].drop(['Prednisolone'],axis=1)
        
    elif treatment == 'no':
        df = df[df['Prednisolone']==0].drop(['Prednisolone'],axis=1)
        
    if missing == 'cc':
        df = df.dropna()
    
    df.reset_index(drop=True, inplace=True)
        
    X, y = df.drop('D90_DTH', axis=1), df[['D90_DTH']]
    
    
    #Split 
    # Splits the data into a train-test set or an untreated train and treated test set 
    if split == 'train-test':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=44)
    
    elif split == 'mixed':
        train = df.loc[df['Prednisolone']==0].drop(['Prednisolone'], axis=1)
        test= df.loc[df['Prednisolone']==1].drop(['Prednisolone'], axis=1)
        train.reset_index()
        test.reset_index()

        X_train = train.drop(['D90_DTH'],axis=1)
        y_train = train['D90_DTH']

        X_test = test.drop(['D90_DTH'],axis=1)
        y_test = test['D90_DTH']
        
        
    return df, X, y, X_train, X_test, y_train, y_test




