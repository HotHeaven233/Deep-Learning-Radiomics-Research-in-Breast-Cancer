# -*- coding:utf-8 -*-  
# @Time     : 2022/05/31 13:45
# @Author   : Chengke
# @File     : confidence.py 
# @version: Python 3.6.13
import numpy as np
import sklearn.metrics as metrics


def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=50):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = metrics.roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))


def bootstrap_pre(clf, X_train, y_train, X_test, y_test, nsamples=50):
    acc_values = []
    sens_values = []
    spec_values = []
    ppv_values = []
    npv_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict(X_test)
        matrix = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        acc = clf.score(X_test, y_test)
        acc_values.append(acc)
        sens_values.append(sens)
        spec_values.append(spec)
        ppv_values.append(ppv)
        npv_values.append(npv)
    return np.percentile(acc_values, (2.5, 97.5)), np.percentile(sens_values, (2.5, 97.5)), np.percentile(spec_values, (
        2.5, 97.5)), np.percentile(ppv_values, (2.5, 97.5)), np.percentile(npv_values, (2.5, 97.5))
