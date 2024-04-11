from dfsubset import *

#Models 

from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import random

# Fitting an XGB model 

def xgb_tuning_cv(X, y):
    param_dist = {
    'max_depth': [random.randint(3, 10)],
    'learning_rate': [random.uniform(0.01, 0.1)],
    'subsample': [random.uniform(0.5, 0.5)],
    'n_estimators':[random.randint(50, 200)]}
    
    #5 Fold Cross-validation 
    xgb_model = XGBClassifier(objective='binary:logistic')
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=1000, cv=5,scoring='roc_auc')
    random_search.fit(X, y)
    #print("Best score: ", random_search.best_score_)
    return random_search

def ebm_tuning_cv(X,y):
    param_dist = {
    'max_bins' :[1024, 4096, 16384, 65536],
    'max_interaction_bins': [8, 16, 32, 64, 128, 256],
    'interactions': [0, 0.25, 0.5, 0.75, 0.95],
    'learning_rate':[0.02, 0.01, 0.005, 0.0025],
    'greediness': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0],
    #'cyclic_progress': [0.0, 0.25, 0.5, 1.0],
    'smoothing_rounds': [0, 50, 100, 200, 500, 1000, 2000, 4000],
    'interaction_smoothing_rounds': [0, 50, 100, 500],
    'min_hessian': [1.0, 0.01, 0.0001, 0.000001]}
    
    ebm = ExplainableBoostingClassifier()
    random_search = RandomizedSearchCV(ebm, param_distributions=param_dist, n_iter=10, cv=5,scoring='roc_auc')
    random_search.fit(X, y)
    return random_search

def xgb_results(subset,treatment,split):
    df = pd.read_csv('stopah.csv')
    df, X, y, X_train, X_test, y_train, y_test = dataset(df,subset,treatment,'cc',split,0.3)
    
    #Hyperparameter tuning
    random_search = xgb_tuning_cv(X, y)
    xgb = XGBClassifier(**random_search.best_params_,objective='binary:logistic')
    xgb.fit(X_train, y_train)
    #Accuracy and AUC 
    
    #Accuracy
    xgb_score = round(xgb.score(X_test,y_test)*100,2)
    
    #AUC
    pred_prob = xgb.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, pred_prob[:,1])
    
    return xgb_score, auc_score

def ebm_results(subset,treatment,split):
    df = pd.read_csv('stopah.csv')
    df, X, y, X_train, X_test, y_train, y_test = dataset(df,subset,treatment,'cc',split,0.3)
    
    random_search = ebm_tuning_cv(X,y)
    #Model fit 
    ebm = ExplainableBoostingClassifier(**random_search.best_params_)
    emb1 = ebm.fit(X_train, y_train)
    score = ebm.score(X_test, y_test)
    auc = roc_auc_score(y_test, emb1.predict_proba(X_test)[:, 1])
    
    return round(score*100,2), round(auc*100,2)
    

