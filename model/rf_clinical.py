# -*- coding:utf-8 -*-  
# @Time     : 2022/04/08 23:41
# @Author   : Chengke
# @File     : rf_clinical.py
# @version: Python 3.6.13


import pandas as pd
import sklearn as skl
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import operator as op
import csv
import sklearn.metrics as metrics
from confidence import *

df = pd.read_csv('data/predict/clinical.csv', encoding='gb18030')
x1, y1 = df.iloc[:, 1:21].values, df.iloc[:, 21].values
np.random.seed(123)

sm = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=1, m_neighbors=5)
x1, y1 = sm.fit_resample(x1, y1)
clf_1 = RandomForestClassifier(n_estimators=300, random_state=0, oob_score=False, bootstrap=False, min_samples_split=3,
                               min_samples_leaf=2, min_weight_fraction_leaf=0.49)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x1, y1, test_size=0.3)
clf_1.fit(X_train_1, y_train_1)

y_predict = clf_1.predict(X_test_1)
y_predict_1 = clf_1.predict_proba(X_test_1)[:, 1]

matrix = skl.metrics.confusion_matrix(y_test_1, y_predict, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
print('Accuracy', clf_1.score(X_test_1, y_test_1))
print('sensitivity: ', tp / (tp + fn))
print('specificity: ', tn / (tn + fp))
print("ppv：", tp / (tp + fp))
print("npv：", tn / (tn + fn))

# confidence interval
auc_CI = bootstrap_auc(clf_1, X_train_1, y_train_1, X_test_1, y_test_1)
acc_CI, sens_CI, spec_CI, ppv_CI, npv_CI = bootstrap_pre(clf_1, X_train_1, y_train_1, X_test_1, y_test_1)
print("auc CI:", auc_CI)
print("acc CI", acc_CI)
print("sens CI", sens_CI)
print("spec CI", spec_CI)
print("ppv CI", ppv_CI)
print("npv CI", npv_CI)
