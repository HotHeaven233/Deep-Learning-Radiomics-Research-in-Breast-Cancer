# -*- coding:utf-8 -*-  
# @Time     : 2022/04/20 2:04
# @Author   : Chengke
# @File     : ROC.py 
# @version: Python 3.6.13
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from rf_T4_T11 import *
from rf_T11 import *
from rf_T4 import *
from rf_clinical import *


def multi_models_roc(names, colors, y_test_S, y_predict_S, dpin=100):
    """
    Output roc graphs of multiple models to one graph

    Args:
        names: list, Multiple model names
        sampling_methods: list, instantiated objects for multiple models

    Returns:
        Returns the image object plt
    """
    plt.figure(figsize=(30, 30), dpi=dpin)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.rc('font', **{'family': 'FZKaTong-M19S'})

    for (name, colorName, y_test, y_predict) in zip(names, colors, y_test_S, y_predict_S):
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predict, pos_label=1)

        plt.plot(fpr, tpr, lw=3, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)), color=colorName)
        plt.plot([0, 1], [0, 1], 'r--', lw=3, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('1-specificity', fontsize=40)
        plt.ylabel('Sensitivity', fontsize=40)
        plt.title('ROC comparision', fontsize=40)
        plt.legend(loc='lower right', fontsize=40)

    return plt


names = ['Classification by clinical information',
         'Clinical information combined DLR(T4)',
         'Clinical information combined DLR(T11)',
         'Clinical information combined DLR(T4&T11)',
         ]
y_test_S = [y_test_1,
            y_test_2,
            y_test_3,
            y_test_4
            ]
y_predict_S = [y_predict_1,   # clinical
               y_predict_2,   # T4
               y_predict_3,   # T11
               y_predict_4,  # T4&T11
               ]
colors = ['black',
          'orange',
          'steelblue',
          'mediumseagreen',
          ]

# ROC curves
train_roc_graph = multi_models_roc(names, colors, y_test_S, y_predict_S)
# train_roc_graph.show()
train_roc_graph.savefig('roc.jpg')
