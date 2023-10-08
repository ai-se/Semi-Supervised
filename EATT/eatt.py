# -*- coding: utf-8 -*-

"""
Effort-aware tri-training for just-in-time defect prediction
author: Wenzhou Zhang
date: 2018/9/24
"""
import random

import numpy as np
import math
from sklearn.svm import *
from sklearn.ensemble import RandomForestClassifier
from itertools import compress
from sklearn.linear_model import LogisticRegression
from EATT.indicator import acc


class EATT():
    """
    Effort-Aware Tri-Training for Semi-Supervised Just-in-Time Defect Prediction
    """

    def __init__(self, estimator=None, bias=True):
        self.estimator = estimator
        self.bias = bias
        self.l_index = self.u_index = None
        self._e_l = np.ones(3) * 0.5
        self._acc_l = np.ones(3) * 0.5
        self._l = np.zeros(3)
        self._acc = np.zeros(3)
        self._e = np.zeros(3)
        self._update = np.array([True, True, True])

    def fit(self, X, Y, effort=None):
        """
        Train the three classifiers by using labeled and unlabeled data.
        :param X:
        :param Y:
        :param effort:
        :return:
        """
        self._init(X, Y)
        Ueffort = effort[self.u_index]
        while np.any(self._update):
            L_idx, L_lab = [], []
            for i, est in enumerate(self.estimator):
                idx = [k for k in range(3) if k != i]
                self._update[i] = False
                self._acc[i] = self._measure_error(i, idx, effort[self.l_index])
                if self._l[i] == 0 and (self._e_l[i] != self._e[i]):
                    self._l[i] = math.floor(self._e[i] / (self._e_l[i] - self._e[i]) + 1)

                if self._acc[i] >= self._acc_l[i]:
                    L_idx.append([])
                    L_lab.append([])
                    continue
                pre = np.vstack((self.estimator[idx[0]].predict(self.X[self.u_index]),
                                 self.estimator[idx[1]].predict(self.X[self.u_index])))
                proba = self.estimator[idx[0]].predict_proba(self.X[self.u_index])[:, 1] + \
                        self.estimator[idx[1]].predict_proba(self.X[self.u_index])[:, 1]
                L_i = pre[0, :] == pre[1, :]

                if (self._e[i] * np.sum(L_i)) < (self._e_l[i] * self._l[i]):
                    self._update[i] = True
                else:
                    if self._e[i] != 0:
                        s = math.ceil((self._e_l[i] * self._l[i] / self._e[i]) - 1)
                    else:
                        s = int(self._l[i])
                    self._subsample(L_i, np.argsort(proba[L_i]/Ueffort[L_i]), s)
                    self._update[i] = True
                L_idx.append(L_i)
                L_lab.append(pre[0, L_i])

            for i, est in enumerate(self.estimator):
                if False == self._update[i]:
                    continue
                s_index = np.array([False] * len(self.l_index))
                s_index[self.u_index] = L_idx[i]
                self.Y[s_index] = L_lab[i]
                s_index[self.l_index] = True
                if self._update[i]:
                    est.fit(self.X[s_index], self.Y[s_index])
                self.Y[self.u_index] = -1

                self._e_l[i] = self._e[i]
                self._acc_l[i] = self._acc[i]
                self._l[i] = np.sum(L_idx[i])

    def _init(self, X, Y):
        if self.bias:
            self.X = np.append(X, np.ones((np.shape(X)[0], 1)), axis=1)
        else:
            self.X = X
        self.Y = np.ravel(Y)
        self.l_index = np.ravel(self.Y != - 1)
        self.u_index = ~self.l_index

        if self.estimator is None:
            self.estimator = [RandomForestClassifier(), LogisticRegression(), svm()]

        # Initialize the classifiers
        for est in self.estimator:
            indices = random.sample(list(compress(range(len(X)), self.l_index)), int(np.sum(self.l_index) * 1))
            est.fit(self.X[indices], self.Y[indices])

    def _measure_error(self, i, idx, effort):
        pre = np.vstack((self.estimator[idx[0]].predict(self.X[self.l_index]),
                         self.estimator[idx[1]].predict(self.X[self.l_index])))
        L_i = pre[0, :] == pre[1, :]
        LY = self.Y[self.l_index]
        pre_same = pre[0, L_i] + 1
        self._e[i] = np.sum(pre[0, L_i] != LY[L_i]) / np.sum(L_i)
        return 1 - acc(pre_same, effort[L_i],  LY[L_i])

    def _subsample(self, L, c, s):
        candate = c[L[c]]
        if s <= 0:
            sel = []
        elif len(candate) > 4 * s:
            sel = random.sample(list(candate[:2 * s]) + list(candate[-2 * s:]), s)
        elif len(candate) > s:
            sel = random.sample(list(candate), s)
        else:
            sel = candate
        L[L] = False
        L[sel] = True

    def predict_R(self, X, effort):
        if self.bias:
            X = np.append(X, np.ones((len(X), 1)), axis=1)
        fin_pre = np.zeros(len(X))
        for i, est in enumerate(self.estimator):
            pre = est.predict_proba(X)[:, 1]
            fin_pre += pre

        return fin_pre/effort

    def predict(self, X):
        if self.bias:
            X = np.append(X, np.ones((len(X), 1)), axis=1)
        pre = np.vstack((self.estimator[0].predict(X), self.estimator[1].predict(X), self.estimator[2].predict(X)))
        return (np.sum(pre, axis=0) > 1) * 1


class svm():

    def __init__(self, bias=True):
        self.w = None
        self.X = None
        self.Y = None
        self.bias = bias
        self.model = SVC()

    def fit(self, X, Y):
        self._init(X, Y)
        self.model.fit(self.X[self.l_index], self.Y[self.l_index])

    def _init(self, X, Y):
        self.n = np.shape(X)[0]
        if self.bias:
            self.X = np.append(X, np.ones((self.n, 1)), axis=1)
        else:
            self.X = X
        self.Y = np.ravel(Y)
        self.l_index = np.ravel(self.Y != - 1)

    def predict(self, X):
        if self.bias:
            X = np.append(X, np.ones((len(X), 1)), axis=1)
        pre = self.model.predict(X)
        return np.ravel(pre)

    def predict_proba(self, X):
        if self.bias:
            X = np.append(X, np.ones((len(X), 1)), axis=1)
        pre_prob = 1 / (1 + np.exp(- self.model.decision_function(X)))
        return np.vstack((1 - pre_prob, pre_prob)).T

