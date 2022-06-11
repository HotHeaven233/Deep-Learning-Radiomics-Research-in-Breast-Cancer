# -*- coding:utf-8 -*-  
# @Time     : 2022/04/08 23:38
# @Author   : Chengke
# @File     : rf_T11.py
# @version: Python 3.6.13


import pandas as pd
import sklearn as skl
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import operator as op
import csv
import sklearn.metrics as metrics
from confidence import *

df = pd.read_csv('data/predict/T11.csv', encoding='gb18030')
x1, y1 = df.iloc[:, 1:31].values, df.iloc[:, 31].values
np.random.seed(123)

sm = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=10, m_neighbors=5)
x1, y1 = sm.fit_resample(x1, y1)
clf_3 = RandomForestClassifier(n_estimators=300, random_state=0, oob_score=False, bootstrap=False, min_samples_split=10,
                               min_samples_leaf=2, min_weight_fraction_leaf=0.48)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(x1, y1, test_size=0.3)
clf_3.fit(X_train_3, y_train_3)
y_predict = clf_3.predict(X_test_3)
y_predict_3 = clf_3.predict_proba(X_test_3)[:, 1]

matrix = skl.metrics.confusion_matrix(y_test_3, y_predict, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
print('Accuracy', clf_3.score(X_test_3, y_test_3))
print('sensitivity: ', tp / (tp + fn))
print('specificity: ', tn / (tn + fp))
print("ppv：", tp / (tp + fp))
print("npv：", tn / (tn + fn))

# confidence interval
auc_CI = bootstrap_auc(clf_3, X_train_3, y_train_3, X_test_3, y_test_3)
acc_CI, sens_CI, spec_CI, ppv_CI, npv_CI = bootstrap_pre(clf_3, X_train_3, y_train_3, X_test_3, y_test_3)
print("auc CI:", auc_CI)
print("acc CI", acc_CI)
print("sens CI", sens_CI)
print("spec CI", spec_CI)
print("ppv CI", ppv_CI)
print("npv CI", npv_CI)

