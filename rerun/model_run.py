from data import *
from ml import *

#Model 
from xgboost import XGBClassifier

#AUC 
from sklearn.model_selection import cross_val_score

#


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
        #score = cross_val_score(training,X,y,cv=10,scoring='score')
        auc = np.mean(cv_aucs)
        
    else:
        #df, X, y, X_train, X_test, y_train, y_test = dataset(subset,death,treatment,split,size,missing)
        model_fit = model.fit(X_train, y_train)
        #score = model_fit.score(X_test, y_test)
        auc = roc_auc_score(y_test, model_fit.predict_proba(X_test)[:, 1])
        
    return auc
        
        
#Fix assignement of model 