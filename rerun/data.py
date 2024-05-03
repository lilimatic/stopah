import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

#MICE
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def dataset(subset,death,treatment,split,size,missing): 
    #
    #-subset: String, determines subset 'Lille','7day','Baseline','Paper'
    #-death: Integer, 30 days or 90 days
    #-treatment: Integer, 1 if treated and else 0
    #-split: String, determines split 'train-test','mix-1','mix-2','cv'
    #-testsize: Float, Size of test set 
    #-missing: String, provides missingness mechanism: 'none','cc','ipw','mice'
    #
    #Read data
    df = pd.read_csv('/Users/lilimatic/stopah.csv')
    
    df['Hepatic.Encephalopathy...Treatment.Day.7..'] = df['Hepatic.Encephalopathy...Treatment.Day.7..'].astype('Int64')

    df['Hepatic.Encephalopathy...Merged'] = df['Hepatic.Encephalopathy...Merged'].astype('Int64')

    
    #Treat values as NA
    for x in df.columns:
        if len(df[df[x] == -2147483648].index) >0:
            ix = df[df[x] == -2147483648].index
            imp = df[[x]].loc[~df.index.isin(ix)].mean()[0]
            df.loc[ix,x] = np.nan
    #Subsets 
    #
    # --- Baseline ---
    #
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
    # --- Lille ---
    #
    Lille_var = ['Age.at.randomisation..calc.','Albumin...Merged','Creatinine..mg.dL....Merged',
             'Bilirubin.Merged','Prothrombin.Time..patient....Merged','Bilirubin.day.7']
    

    
    ### Subset 
    
    if subset == 'Lille':
        df = df[Lille_var+['Prednisolone',f'D{death}_DTH']]
        
        df['Renal Insufficency'] = (df['Creatinine..mg.dL....Merged'].loc[:] > 1.3).astype('float')
        df                       = df.drop(['Creatinine..mg.dL....Merged'],axis=1)
        df['Bilirubin.day.7']    =  df['Bilirubin.Merged'] - df['Bilirubin.day.7'] 
         
    elif subset == '7day':
        df                       = df[sevenday+['Prednisolone',f'D{death}_DTH']]
    elif subset == 'baseline':
        df                       = df[baseline+['Prednisolone',f'D{death}_DTH']]
    elif subset == 'full':
        df                       = df[sevenday+baseline+['Prednisolone',f'D{death}_DTH']]
    
    #Treatment 
    if treatment == 1:
        df                       = df[df['Prednisolone']==1].drop(['Prednisolone'],axis=1)
        
    elif treatment == 0:
        df                       = df[df['Prednisolone']==0].drop(['Prednisolone'],axis=1)
    
    elif treatment == 99:
        df = df
        
    ### MISSING VALUE MANAGEMENT 
    
    #No missing value management for tree based models 
    if missing == 'none':
        df =df
    #
    # Complete-Case analysis 
    #
    elif missing == 'cc':
        # Complete-case analysis
        # 
        df = df.dropna() 
        #
        # HOT-ONE encoding 
        #
        cat_var = [x for x in df.columns if df[x].dtype != 'float64']
        df[cat_var] = df[cat_var].astype('Int64')
        #
        #Add hot-one encoded categories 
        #
        for cat in cat_var:
            if cat != ('D28_DTH' or 'D90_DTH'):
                new = cat + '_2'
                df[new] = df[cat].apply(lambda x: 1 if x== 0 else 0 )
                df[new] = df[new].astype('Int64')
        
    #
    # Inverse-probability-weighting in complete-cases
    elif (missing == 'ipw' or missing == 'mice'):
        if subset == 'full':
            s = ['Hepatic.Encephalopathy...Merged','Hepatic.Encephalopathy...Treatment.Day.7..','Gastrointestinal.Bleed.since.the.last.visit.Gastrointestinal.bleed...and.Choose..Treatment.Day.7..']
            df[s] = df[s].astype('Int64')
            imp = SimpleImputer(strategy="most_frequent")
            for x in s:
                df[x] = imp.fit_transform(df[x].values.reshape(-1, 1))
                df[x] = df[x].astype('Int64')
        if subset == 'baseline':
            s = 'Hepatic.Encephalopathy...Merged'
            df[s] = df[s].astype('Int64') 
            imp = SimpleImputer(strategy="most_frequent")
            df[s] = imp.fit_transform(df[s].values.reshape(-1, 1))
            df[s] = df[s].astype('Int64')
        elif subset == '7day':
            s = ['Gastrointestinal.Bleed.since.the.last.visit.Gastrointestinal.bleed...and.Choose..Treatment.Day.7..','Hepatic.Encephalopathy...Treatment.Day.7..']
            df[s] = df[s].astype('Int64')
            imp = SimpleImputer(strategy="most_frequent")
            for x in s:
                df[x] = imp.fit_transform(df[x].values.reshape(-1, 1))
                df[x] = df[x].astype('Int64')
        #
        #IPW
        #
        if missing == 'ipw':
            #cat_var = [x for x in df.co if (df[x].dtypes  == 'Int64' or df[x].dtypes  == 'int64') ]
            cat_var = [x for x in df.columns if df[x].dtype != 'float64']
            df[cat_var] = df[cat_var].astype('Int64')
            #Add hot-one encoded categories 
            for cat in cat_var:
                if cat != ('D28_DTH' or 'D90_DTH'):
                    new = cat + '_2'
                    df[new] = df[cat].apply(lambda x: 1 if x== 0 else 0 )
                    df[new] = df[new].astype('Int64')
            #
            #Inverse-Probabilty-Weighting
            #
            probs = list(1 -(df.isnull().sum().values / len(df)))
            df = df.div(probs, axis=1)
            df =  df.dropna()
            df = df.astype('float64')
            
        #
        #MICE
        #
        elif missing == 'mice':
            mice = IterativeImputer(max_iter=10, random_state=0)
            df = pd.DataFrame(mice.fit_transform(df), columns=df.columns)
            cat_var = [x for x in df if (df[x].dtypes  == 'Int64' or df[x].dtypes  == 'int64') ]
            df[cat_var] = df[cat_var].astype('Int64')

    elif missing == 'none':
        df =df 
        
        
    df.reset_index(drop=True, inplace=True)
        
    X, y = df.drop(f'D{death}_DTH', axis=1), df[[f'D{death}_DTH']]
    
    
    #Split 
    # Splits the data into a train-test set or an untreated train and treated test set 
    if split == 'train-test':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=44)
        
    # MIX-1 train on non-treated test on treated
    elif split == 'mix-1':
        train = df.loc[df['Prednisolone']==0].drop(['Prednisolone'], axis=1)
        test= df.loc[df['Prednisolone']==1].drop(['Prednisolone'], axis=1)
        train.reset_index()
        test.reset_index()

        X_train = train.drop([f'D{death}_DTH'],axis=1)
        y_train = train[f'D{death}_DTH']

        X_test = test.drop([f'D{death}_DTH'],axis=1)
        y_test = test[f'D{death}_DTH']
        
    # MIX-2 train on non-treated test on treated
    elif split == 'mix-2':
        train = df.loc[df['Prednisolone']==1].drop(['Prednisolone'], axis=1)
        test= df.loc[df['Prednisolone']==0].drop(['Prednisolone'], axis=1)
        train.reset_index()
        test.reset_index()

        X_train = train.drop([f'D{death}_DTH'],axis=1)
        y_train = train[f'D{death}_DTH']

        X_test = test.drop([f'D{death}_DTH'],axis=1)
        y_test = test[f'D{death}_DTH']
    
    
    if split != 'cv':
        return df, X, y, X_train, X_test, y_train, y_test
    else: 
        return df, X, y, X, X, y, y
    

