# -*- coding:utf-8 -*-  
# @Time     : 2022/04/08 23:10
# @Author   : Chengke
# @File     : rf_T4_T11.py
# @version: Python 3.6.13

import pandas as pd
import sklearn as skl
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import miceforest as mf
import csv
import sklearn.metrics as metrics
from confidence import *

df = pd.read_csv('data/predict/T4&T11.csv', encoding='gb18030', header=0)
x1, y1 = df.iloc[:, 1:41].values, df.iloc[:, 41].values

np.random.seed(123)

sm = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=10, m_neighbors=5)
x1, y1 = sm.fit_resample(x1, y1)
clf_4 = RandomForestClassifier(n_estimators=300, random_state=0, oob_score=False, bootstrap=False, min_samples_split=3,
                               min_samples_leaf=2, min_weight_fraction_leaf=0.4)
X_train_4, X_test_4, y_train_4, y_test_4, index_train, index_test = train_test_split(x1, y1, range(len(x1)),
                                                                                     test_size=0.3)
clf_4.fit(X_train_4, y_train_4)
y_predict_4 = clf_4.predict_proba(X_test_4)[:, 1]
index = [i + 2 for i in index_test]
original_test_index = []
for i in index:
    if i <= len(t) + 1:
        original_test_index.append(i)
print("Test set index after oversampling", index)
print("Test set index before oversampling：", original_test_index)

# Results report
y_predict = clf_4.predict(X_test_4)
matrix = skl.metrics.confusion_matrix(y_test_4, y_predict, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
print('Accuracy', clf_4.score(X_test_4, y_test_4))
print('sensitivity: ', tp / (tp + fn))
print('specificity: ', tn / (tn + fp))
print("ppv：", tp / (tp + fp))
print("npv：", tn / (tn + fn))



# confidence interval
auc_CI = bootstrap_auc(clf_4, X_train_4, y_train_4, X_test_4, y_test_4)
acc_CI, sens_CI, spec_CI, ppv_CI, npv_CI = bootstrap_pre(clf_4, X_train_4, y_train_4, X_test_4, y_test_4)
print("auc CI:", auc_CI)
print("acc CI", acc_CI)
print("sens CI", sens_CI)
print("spec CI", spec_CI)
print("ppv CI", ppv_CI)
print("npv CI", npv_CI)
