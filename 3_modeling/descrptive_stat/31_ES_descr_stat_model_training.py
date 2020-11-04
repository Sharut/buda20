#########################################################
##                                                     ##
##                      importing packages             ##
##                                                     ##
#########################################################
import joblib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

#reading DF
es_data_tweet_consist = pd.read_pickle('NLP/PAN20/data/es_data_tweet_consist.pkl')

#########################################################
##                                                     ##
##                    model training                   ##
##                                                     ##
#########################################################

#######
#
# benchmark with default setups on half training data
#

xgb_def = xgb.XGBClassifier()
xgb_def.fit(es_data_tweet_consist.iloc[50:,2:],
            es_data_tweet_consist.iloc[50:,1:2])

#train data metrics
print(classification_report(es_data_tweet_consist.iloc[:50,1:2],
                            xgb_def.predict(es_data_tweet_consist.iloc[:50,2:])))
#test data metrics
print(classification_report(es_data_tweet_consist.iloc[50:,1:2],
                            xgb_def.predict(es_data_tweet_consist.iloc[50:,2:])))


########################
##
## hyperpar search
##

#scoring metrics
f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)
scoring = {'F1': f1_scorer, 'Accuracy': acc_scorer}

#defining search space
params = {
        'min_child_weight': [2, 3, 4, 5],
        'gamma': [0, 1, 2, 4],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'colsample_bynode': [0.8, 0.9, 1],
        'max_depth': [2, 3, 4],
        'learning_rate' : [0.3, 0.2, 0.1],
        'reg_alpha':[0.1, 0.3, 0.7],
        'n_estimators' : [100, 150, 200]
        }


# grid search with CV
xgb_cv_clf = xgb.XGBClassifier()
xgb_cv = GridSearchCV(xgb_cv_clf,
                  param_grid=params,
                  scoring=scoring,
                  refit = "Accuracy",
                  return_train_score=True,
                  cv=5,
                  verbose=1,
                  n_jobs = -1)
xgb_cv.fit(es_data_tweet_consist.iloc[:,2:], es_data_tweet_consist.iloc[:,1:2])

# performance metrics
print(xgb_cv.best_score_, xgb_cv.best_params_)

print(classification_report(es_data_tweet_consist.iloc[:,1:2],
                            xgb_cv.best_estimator_.predict(es_data_tweet_consist.iloc[:,2:])))

joblib.dump(xgb_cv.best_estimator_, "NLP/PAN20/models/tweetconsistence_xgboost_es_v2")