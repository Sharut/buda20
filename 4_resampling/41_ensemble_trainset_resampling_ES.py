#########################################################
##                                                     ##
##                      importing packages             ##
##                                                     ##
#########################################################
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import xgboost as xgb

#########################################################
##                                                     ##
##                       reading data                  ##
##                                                     ##
#########################################################
loc_data = '' #location of files has to be set

#loading feature DF-s
data_es_v1=pd.read_csv(loc_data + 'clean_es_data_v1.tsv', delimiter='\t',
                      encoding='utf-8')
data_es_v2=pd.read_csv(loc_data + 'clean_es_data_v2.tsv', delimiter='\t',
                      encoding='utf-8')
data_es_tw_cons=pd.read_csv(loc_data + 'es_data_tweet_consist.tsv', delimiter='\t',
                      		encoding='utf-8')

# best hyperparameters found for individual models
xgb_pl = Pipeline([('vect', TfidfVectorizer(min_df=8, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('xgb', xgb.XGBClassifier(colsample_bytree= 0.7, eta= 0.3, max_depth= 6, n_estimators= 200, subsample= 0.6))])

lr_pl=Pipeline([('vect', TfidfVectorizer(min_df=9, ngram_range=(2,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('lr', LogisticRegression(C=100, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0))])

rf_pl=Pipeline([('vect', TfidfVectorizer(min_df=3, ngram_range=(1,2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
                ('rf', RandomForestClassifier(n_estimators=100, min_samples_leaf=8, criterion='gini'))])

svm_pl=Pipeline([('vect', TfidfVectorizer(ngram_range=(2,2), min_df=8, sublinear_tf=True, use_idf=True, smooth_idf=True)),
                ('rf', SVC(C=10, kernel='linear', verbose=False, probability=True))])

xgb_twc = xgb.XGBClassifier(colsample_bynode = 0.8,
                            colsample_bytree = 0.8,
                            gamma = 4,
                            learning_rate = 0.3,
                            max_depth = 3,
                            min_child_weight = 5,
                            n_estimators = 100,
                            reg_alpha = 0.3,
                            subsample = 0.8)

X=data_es_v1["Tweets"]
y=data_es_v1['spreader']

#########################################################
##                                                     ##
## constructing dev & train feature DF-s for ensemble  ##
##                                                     ##
#########################################################
#train set
cv = StratifiedKFold(5, shuffle=True)
results = []
for train_index, test_index in cv.split(X, y):
    preds = pd.DataFrame()
    y_train, y_test = y[train_index], y[test_index]
    preds['y_truth'] = y[test_index].values
    #LR
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    lr_pl.fit(X_train,y_train)
    preds["lr"] = lr_pl.predict_proba(X_test)[:,1]

    #SVM
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    svm_pl.fit(X_train,y_train)
    preds["svm"] = svm_pl.predict_proba(X_test)[:,1]

    #RF
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    rf_pl.fit(X_train,y_train)
    preds["rf"] = rf_pl.predict_proba(X_test)[:,1]

    #XGB
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    xgb_pl.fit(X_train,y_train)
    preds["xgb"] = xgb_pl.predict_proba(X_test)[:,1]

    #XGB on tweets
    X_train, X_test = data_es_tw_cons.iloc[list(train_index), 2: ], data_es_tw_cons.iloc[list(test_index), 2: ]
    xgb_twc.fit(X_train,y_train)
    preds["xgb_tw"] = xgb_twc.predict_proba(X_test)[:,1]

    results.append(preds)

result_es=pd.concat(results)
result_es.to_csv('NLP/PAN20/data/es_ensemble_train.tsv', sep='\t', index=False)

#dev set
cv = StratifiedKFold(5, shuffle=True)
results = []
for train_index, test_index in cv.split(X, y):
    preds = pd.DataFrame()
    y_train, y_test = y[train_index], y[test_index]
    preds['y_truth'] = y[test_index].values
    #LR
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    lr_pl.fit(X_train,y_train)
    preds["lr"] = lr_pl.predict_proba(X_test)[:,1]

    #SVM
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    svm_pl.fit(X_train,y_train)
    preds["svm"] = svm_pl.predict_proba(X_test)[:,1]

    #RF
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    rf_pl.fit(X_train,y_train)
    preds["rf"] = rf_pl.predict_proba(X_test)[:,1]

    #XGB
    X_train, X_test = data_es_v1["Tweets"][train_index], data_es_v1["Tweets"][test_index]
    xgb_pl.fit(X_train,y_train)
    preds["xgb"] = xgb_pl.predict_proba(X_test)[:,1]

    #XGB on tweets
    X_train, X_test = data_es_tw_cons.iloc[list(train_index), 2: ], data_es_tw_cons.iloc[list(test_index), 2: ]
    xgb_twc.fit(X_train,y_train)
    preds["xgb_tw"] = xgb_twc.predict_proba(X_test)[:,1]

    results.append(preds)
result_es_v2=pd.concat(results)
result_es_v2.to_csv('NLP/PAN20/data/es_ensemble_dev.tsv', sep='\t', index=False)
