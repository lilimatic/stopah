from data import *

# Basic 
import random

# Model 
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier


# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV


# Metrics 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


# 1. Hyperparemter tuning 


# XGB 5-Fold-CV tuning
def xgb_tuning_cv(X,y):
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

# EBM 5-Fold-CV tuning

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


def model(model,ttype,subset,death,treatment,split,size,missing): 
    #
    #-model: String, model to be trained: 'xgb','elasticnet','' 
    #-ttype: String, determines model to be trained 
    #-subset: String, determines subset: 'Lille','7day','Baseline','Paper'
    #-death: Integer, 30 days or 90 days
    #-treatment: Integer, 1 if treated and else 0
    #-split: String, determines split: 'train-test','mixed','cv'
    #-testsize: Float, Size of test set 
    #-missing: String, provides missingness mechanism: 'none','cc','ipw','mice'
    #
    # MODEL TRAINING
    #
    # Train XGB
    #
    # Read data
    df, X, y, X_train, X_test, y_train, y_test = dataset(subset,death,treatment,split,size,missing)
    #
    if model == 'xgb' :
        random_search = xgb_tuning_cv(X, y)
        training = XGBClassifier(**random_search.best_params_,objective='binary:logistic')
    #
    # Train EBM
    #
    elif model == 'gam':
        random_search = ebm_tuning_cv(X,y)
        training = ExplainableBoostingClassifier(**random_search.best_params_)
    #
    #Model Training
    #  
    if ttype == 'cv':
        #df, X,y = dataset(subset,death,treatment,split,size,missing)
        cv_aucs = cross_val_score(training,X,y,cv=10,scoring='roc_auc')
        score = cross_val_score(training,X,y,cv=10,scoring='score')
        auc = np.mean(cv_aucs)
        
    else:
        #df, X, y, X_train, X_test, y_train, y_test = dataset(subset,death,treatment,split,size,missing)
        model_fit = model.fit(X_train, y_train)
        score = model_fit.score(X_test, y_test)
        auc = roc_auc_score(y_test, model_fit.predict_proba(X_test)[:, 1])
        
    return auc
        
        
#Fix assignement of model 