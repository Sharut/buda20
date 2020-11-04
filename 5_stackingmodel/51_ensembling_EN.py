#########################################################
##                                                     ##
##                      importing packages             ##
##                                                     ##
#########################################################
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb

#########################################################
##                                                     ##
##          searching best stacking model              ##
##                                                     ##
#########################################################
#load DF-s
result_en = pd.read_csv('NLP/PAN20/data/en_ensemble_train.tsv', , delimiter='\t',
                          encoding='utf-8')
result_en_v2 = pd.read_csv('NLP/PAN20/data/en_ensemble_dev.tsv', , delimiter='\t',
                          encoding='utf-8')

# MEAN
print(classification_report(result_en["y_truth"], result_en.iloc[:,1:].mean(axis=1)>0.5))
print(confusion_matrix(result_en["y_truth"], result_en.iloc[:,1:].mean(axis=1)>0.5))

print(classification_report(result_en_v2["y_truth"], result_en_v2.iloc[:,1:].mean(axis=1)>0.5))
print(confusion_matrix(result_en_v2["y_truth"], result_en_v2.iloc[:,1:].mean(axis=1)>0.5))

# MAJORITY
print(classification_report(result_en["y_truth"], (result_en.iloc[:,1:]>0.5).mean(1)>0.5))
print(confusion_matrix(result_en["y_truth"], (result_en.iloc[:,1:]>0.5).mean(1)>0.5))

print(classification_report(result_en_v2["y_truth"], (result_en_v2.iloc[:,1:]>0.5).mean(1)>0.5))
print(confusion_matrix(result_en_v2["y_truth"], (result_en_v2.iloc[:,1:]>0.5).mean(1)>0.5))

# LOGREG
acc_scorer = make_scorer(accuracy_score)
scoring = {'Accuracy': acc_scorer}
params = {'solver' : ['saga'],
        'penalty' : ['elasticnet'],
        'C' : [0, 0.1, 0.2, 0.4, 0.7, 0.9, 1, 1.2, 1.5, 2],
        'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

logreg_clf = LogisticRegression()
logreg = GridSearchCV(logreg_clf,
                  param_grid=params,
                  scoring=scoring,
                  refit = "Accuracy",
                  return_train_score=True,
                  cv=5,
                  verbose=1,
                  n_jobs = -1)
logreg.fit(result_en.iloc[:,1:], result_en["y_truth"])

#performance metrics and hyperpars
print(logreg.best_score_, logreg.best_params_, "\n",
      logreg.best_estimator_, "\n",
      logreg.best_estimator_.coef_, logreg.best_estimator_.intercept_)

print(classification_report(result_en["y_truth"],
                            logreg.predict(result_en.iloc[:,1:])))
print(confusion_matrix(result_en["y_truth"],
                            logreg.predict(result_en.iloc[:,1:])))

print(classification_report(result_en_v2["y_truth"],
                            logreg.predict(result_en_v2.iloc[:,1:])))
print(confusion_matrix(result_en_v2["y_truth"],
                            logreg.predict(result_en_v2.iloc[:,1:])))

# LINREG
params = {'alpha' : [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15]}
linreg_clf = RidgeClassifier()
linreg = GridSearchCV(linreg_clf,
                  param_grid=params,
                  scoring=scoring,
                  refit = "Accuracy",
                  return_train_score=True,
                  cv=5,
                  verbose=1,
                  n_jobs = -1)
linreg.fit(result_en.iloc[:,1:], result_en["y_truth"])
print(linreg.best_score_, linreg.best_params_, "\n",
      linreg.best_estimator_, "\n",
      linreg.best_estimator_.coef_, linreg.best_estimator_.intercept_)

#performance metrics
print(classification_report(result_en["y_truth"],
                            linreg.predict(result_en.iloc[:,1:])))

print(classification_report(result_en_v2["y_truth"],
                            linreg.predict(result_en_v2.iloc[:,1:])))

# save out logistic regression for aggregation
joblib.dump(logreg, 'NLP/PAN20/models/ensemble_en_logreg')