# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import random

from sklearn.model_selection import ShuffleSplit
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer
import scipy.io as sio

datasets = ['bugzilla', 'columba', 'jdt', 'mozilla', 'platform', 'postgres']


key = ['ns', 'nm', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'pd',\
       'npt', 'exp', 'rexp', 'sexp', 'bug']


class preprocessing():

    def __init__(self, dataset=None, key=None, rate=0.1, label='bug'):
        """
        :param dataset: the name of file
        :param key:     the names of change measures(column)
        :param label    the name of label, default 'bug'
        """
        self.rate = rate
        self.dataset = dataset
        self.key = key
        self.label_name = label
        self.data = None
        self.data_array = None

    def _data_init(self):
        self.data = pd.read_csv('jit_datasets/'+self.dataset+'.csv', index_col='commitdate')
        self.data.index = pd.to_datetime(self.data.index, format='%Y/%m/%d %H:%M').strftime('%Y-%m')
        self.label = self.data.pop(self.label_name)
        if self.key is None:
            self.key = list(self.data.keys())
        del self.data['transactionid']

    def log_trans(self):
        self._data_init()
        transformer = FunctionTransformer(np.log1p)
        X = self.data.values
        y = self.label.values
        X = self.data_array = transformer.transform(X)
        sio.savemat("clean_data/" + self.dataset, {'X': X, 'y': y})

    def time_wise_idx(self):
        """
        :return:
        """
        # get the index of time-wise validation
        self._data_init()
        self.data['index'] = pd.Series(range(len(self.data)), index=self.data.index)
        time_group = np.sort(self.data.index.unique().values)
        unlabel_index, test_index, label_index = [], [], []
        for i, slice in enumerate(time_group):
            if len(time_group) - i <= 5:
                break
            labeled_idx = self.data.loc[[slice, time_group[i+1]], 'index'].values
            unlabel_idx = self.data.loc[[time_group[i+2], time_group[i+3]], 'index'].values
            test_index.append(self.data.loc[[time_group[i+4], time_group[i+5]], 'index'].values)

            y_labeled = self.label[labeled_idx]
            bug_N = min(np.count_nonzero(y_labeled == 1), np.count_nonzero(y_labeled == 0))
            lab_rand_resam = random.sample(list(np.where(y_labeled == 0)[0]), bug_N) + \
                             random.sample(list(np.where(y_labeled == 1)[0]), bug_N)

            label_index.append(labeled_idx[lab_rand_resam])


            y_unlabeled = self.label[unlabel_idx]
            bug_N = min(np.count_nonzero(y_unlabeled == 1), np.count_nonzero(y_unlabeled == 0))
            ubl_rand_resam = random.sample(list(np.where(y_unlabeled == 0)[0]), bug_N) + \
                             random.sample(list(np.where(y_unlabeled == 1)[0]), bug_N)

            unlabel_index.append(unlabel_idx[ubl_rand_resam])

        np.savez("index/time_wise/" + self.dataset + ".npz", label_idx=label_index, unlabel_idx=unlabel_index, test_idx=test_index)

    def cross_vad_idx(self, rate=0.1):
        data = sio.loadmat("clean_data/" + self.dataset)
        ss = ShuffleSplit()
        X = data['X']
        y = data['y'][0]
        train_idx, test_idx, label_idx = [], [], []
        for i in range(10):
            train_arr, test_arr, label_arr = [], [], []
            for train_index, test_index in ss.split(X):
                test_arr.append(test_index)
                # resample
                y_train = y[train_index]
                bug_N = np.count_nonzero(y_train == 1)
                rand_resam = random.sample(list(np.where(y_train == 0)[0]), bug_N) + list(np.where(y_train == 1)[0])
                train_arr.append(train_index[rand_resam])
                # choose the changes which will be viewed as labeled samples.
                y_train = y_train[rand_resam]
                labeled_N = np.shape(y_train)[0] * rate
                random_labeled_points = random.sample(list(np.where(y_train == 0)[0]), int(labeled_N / 2)) + \
                                        random.sample(list(np.where(y_train == 1)[0]), int(labeled_N / 2))
                label_arr.append(random_labeled_points)
            train_idx.append(train_arr)
            test_idx.append(test_arr)
            label_idx.append(label_arr)
        np.savez("index/cross_vad/"+str(rate)+"/"+self.dataset+".npz", train_idx=train_idx, test_idx=test_idx, label_idx=label_idx)
        # train_idx, test_idx是针对所有数据集上的索引(训练样本已经过重采样) y[train_idx[i]]
        # label_idx是针对训练数据集上的索引，随机标记了比例为rate的样本     y_train[label_idx[i]]


for k in range(6):
    preproc = preprocessing(dataset=datasets[k])
    preproc.time_wise_idx()
    # preproc.log_trans()
    # preproc.cross_vad_idx(rate=0.2)

