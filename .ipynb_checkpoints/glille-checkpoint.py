import os
import pandas as pd
import numpy as np

#Missing value analysis 
import missingno as msno

#Models 
from sklearn.ensemble import GradientBoostingClassifier
from pygam import * 

#Analytics 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV

from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score 



df = pd.read_csv('stopah.csv')

Lille_var = ['Age.at.randomisation..calc.','Albumin...Merged','Creatinine..mg.dL....Merged',
             'Bilirubin.Merged','Prothrombin.Time..patient....Merged','Bilirubin.day.7','D28_DTH','Prednisolone']

df = df[Lille_var]

#How should we treat those ? 

for x in df.columns:
    if len(df[df[x] == -2147483648].index) >0:
        #print(x)
        ix = df[df[x] == -2147483648].index
        imp = df[[x]].loc[~df.index.isin(ix)].mean()[0]
        df.loc[ix,x] = np.nan

#stopah.min() check for more 

#Phillip Code 

df['Renal Insufficency'] = (df['Creatinine..mg.dL....Merged'].loc[:] > 1.3).astype('float')

df = df.dropna()

df1 = df[['Creatinine..mg.dL....Merged']]

df = df.drop(['Creatinine..mg.dL....Merged'],axis=1)



df['Bilirubin.day.7'] = df['Bilirubin.day.7'] - df['Bilirubin.Merged']

df0 = df[df['Prednisolone']==0].drop(['Prednisolone'],axis=1)

df = df[df['Prednisolone']==1].drop(['Prednisolone'],axis=1)

X, y = df.drop('D28_DTH', axis=1), df[['D28_DTH']]

X0, y0 = df0.drop('D28_DTH', axis=1), df0[['D28_DTH']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)

X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, random_state=1,test_size=0.2)

#Build-in hyperparameters

gam  = LogisticGAM().fit(X_train.values,y_train.values)

gam_cv = LogisticGAM().gridsearch(X.to_numpy(), y.to_numpy())
gam_cv0 = LogisticGAM().gridsearch(X0.to_numpy(), y0.to_numpy())