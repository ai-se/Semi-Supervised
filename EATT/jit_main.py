# -*- coding: utf-8 -*-

"""
Perform 10 times 10-fold cross-validation and time-wise cross-validation.
author: Wenzhou Zhang
date: 2018/9/23
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from EATT.indicator import *
from eatt import EATT
import sklearn.metrics as sm
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# datasets
datasets = ['bugzilla', 'columba', 'jdt', 'mozilla', 'platform', 'postgres']


def cross_validation():
    for labeled_rate in [0.1, 0.2]:
        print("cross_validation at labeled rate:", labeled_rate)
        indicator = ['pre', 'rec', 'f1', 'acc', 'opt']
        algorithm = 'EATT'
        for k in range(0, 6):
            proj_score = []
            try:
                for ind in indicator:
                    proj_score.append(pd.read_csv("score/"+str(labeled_rate)+"/"+datasets[k]+"/"+ind+".csv", index_col=0))
            except:
                for ind in indicator:
                    proj_score.append(pd.DataFrame())

            data_ori = pd.read_csv('jit_datasets/' + datasets[k] + '.csv')
            effort = ((data_ori['ld']+data_ori['la'])*data_ori['lt']*data_ori['nf']).values/2 + 1
            data = sio.loadmat("clean_data/" + datasets[k])
            X = data['X']
            X = np.delete(X, [4, 5], 1)    		# delete 'la' and 'ld'
            X = np.delete(X, [1, 10], 1)        # delete 'nm' and 'rexp'
            y = data['y'][0]
            idx = np.load("index/cross_vad/" + str(labeled_rate) + '/' + datasets[k] + '.npz')
            train_idx, test_idx, label_idx = idx['train_idx'], idx['test_idx'], idx['label_idx']

            curr_vad = 0

            for i in range(10):
                train_idx_curr, test_idx_curr, label_idx_curr = train_idx[i], test_idx[i], label_idx[i]
                for train_index, test_index, label_index in zip(train_idx_curr, test_idx_curr, label_idx_curr):
                    X_train, y_train_t = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]
                    y_train = np.ones(len(y_train_t)) * -1
                    y_train[label_index] = y_train_t[label_index]
                    # y_train = y_train_t

                    tt = eval(algorithm + '()')
                    tt.fit(X_train, y_train, effort[train_index])

                    pre = tt.predict(X_test)
                    proj_score[0].loc[curr_vad, algorithm] = sm.precision_score(y_test, pre)
                    proj_score[1].loc[curr_vad, algorithm] = sm.recall_score(y_test, pre)
                    proj_score[2].loc[curr_vad, algorithm] = sm.f1_score(y_test, pre)
                    
                    pre = tt.predict_R(X_test, effort[test_index])

                    proj_score[3].loc[curr_vad, algorithm] = acc(pre, effort[test_index], y_test)
                    proj_score[4].loc[curr_vad, algorithm] = P_opt(pre, effort[test_index], y_test)

                    curr_vad += 1
                    print('dataset:', datasets[k], '****** validation count:', curr_vad)
            for i, ind in enumerate(indicator):
                proj_score[i].to_csv("score/"+str(labeled_rate)+"/"+datasets[k]+"/"+ind+".csv")
                print(ind, proj_score[i].mean().values)


cross_validation()


def timewise_validation():
    indicator = ['pre', 'rec', 'f1', 'acc', 'opt']
    algorithm = 'EATT'
    print("time_wise_validation")
    for k in range(0, 6):
        proj_score = []
        try:
            for ind in indicator:
                proj_score.append(pd.read_csv("score/tw/"+datasets[k]+"/"+ind+".csv", index_col=0))
        except:
            for ind in indicator:
                proj_score.append(pd.DataFrame())

        data_ori = pd.read_csv('jit_datasets/' + datasets[k] + '.csv')
        effort = ((data_ori['ld']+data_ori['la'])*data_ori['lt']*data_ori['nf']).values/2 + 1
        data = sio.loadmat("clean_data/" + datasets[k])
        X = data['X']
        X = np.delete(X, [4, 5], 1)        # delete 'la' and 'ld'
        X = np.delete(X, [1, 10], 1)       # delete 'nm' and 'rexp'
        y = data['y'][0]

        idx = np.load("index/time_wise/" + datasets[k] + '.npz')
        label_idx, unlabel_idx, test_idx = idx['label_idx'], idx['unlabel_idx'], idx['test_idx']
        curr_vad = 0

        for label_index, unlabel_index, test_index in zip(label_idx, unlabel_idx, test_idx):
            X_train, y_train_t = X[list(label_index)+list(unlabel_index)], y[list(label_index)+list(unlabel_index)]
            X_test, y_test = X[test_index], y[test_index]
            y_train = np.ones(np.shape(y_train_t)) * -1
            y_train[:len(label_index)] = y_train_t[:len(label_index)]
            # y_train = y_train_t

            tt = eval(algorithm+'()')
            tt.fit(X_train, y_train, effort[list(label_index)+list(unlabel_index)])

            pre = tt.predict(X_test)
            proj_score[0].loc[curr_vad, algorithm] = sm.precision_score(y_test, pre)
            proj_score[1].loc[curr_vad, algorithm] = sm.recall_score(y_test, pre)
            proj_score[2].loc[curr_vad, algorithm] = sm.f1_score(y_test, pre)

            pre = tt.predict_R(X_test, effort[test_index])

            proj_score[3].loc[curr_vad, algorithm] = acc(pre, effort[test_index], y_test)
            proj_score[4].loc[curr_vad, algorithm] = P_opt(pre, effort[test_index], y_test)

            curr_vad += 1
            print('dataset:', datasets[k], '****** validation count:', curr_vad)

        for i, ind in enumerate(indicator):
            proj_score[i].to_csv("score/tw/"+datasets[k]+"/"+ind+".csv")
            print(ind, proj_score[i].mean().values)


timewise_validation()
