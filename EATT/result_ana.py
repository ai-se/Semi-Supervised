# -*- coding: utf-8 -*-

"""
Examine the statistically significant difference
author: Wenzhou Zhang
date: 2018/9/24
"""

import numpy as np
import pandas as pd
import scipy.stats


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def cliff_delta_value(X, Y):
    """
    Calculates Cliff's Delta function, a non-parametric effect magnitude
    test. See: http://revistas.javeriana.edu.co/index.php/revPsycho/article/viewFile/643/1092
    for implementation details.
    :param X:
    :param Y:
    :return:
    """

    # calculate length of vetors.
    lx = len(X)
    ly = len(Y)

    # comparison matrix. First dimension represnt elements in X, the second elements in Y
    # Values calculated as follows:
    # mat(i,j) = 1 if X(i) > Y(j), zero if they are equal, and -1 if X(i) < Y(j)
    mat = np.zeros((lx, ly))

    # perform all the comparisons.
    for i in range(lx):
        for j in range(ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    # calculate delta.
    delta = {}
    value = np.abs(np.sum(mat) / (lx * ly))
    if value >= 0.474:
        delta['magnitude'] = 'Large'
    elif value >= 0.33:
        delta['magnitude'] = 'Medium'
    elif value >= 0.147:
        delta['magnitude'] = 'Small'
    else:
        delta['magnitude'] = 'Negligible'

    delta['estimate'] = value

    return delta


def get_magnitude(data):
    """
    :data:
    :return:
    """
    algos = data.columns
    res = pd.Series(index=algos)
    base = data[algos[-1]]
    for alg in algos[:-1]:
        res.loc[alg] = cliff_delta_value(data[alg], base)['magnitude']
    return res


def stat_info():
    """
    Calculate the mean score and significant difference of difference algorithms.
    :return:
    """
    datasets = ['bugzilla', 'columba', 'jdt', 'mozilla', 'platform', 'postgres']
    algorithms = ['EATT']
    for setting in ['tw']:
        for ind in ['pre', 'rec', 'f1', 'acc', 'opt']:
            avg_score = pd.DataFrame(index=datasets, columns=algorithms)
            sig_score = pd.DataFrame(index=datasets, columns=algorithms)
            for dataset in datasets:
                result = pd.read_csv("score/"+str(setting)+"/"+dataset+"/"+ind+".csv", index_col=0)
                avg_score.loc[dataset] = result.mean()
                sig_score.loc[dataset] = get_magnitude(result)
            avg_score.loc['average'] = avg_score.mean()
            formater = lambda x: "%0.4f" % x
            avg_score = avg_score.applymap(formater)

            avg_score.to_csv("score/stat_info/"+str(setting)+"/"+ind+'.csv')
            sig_score.to_csv("score/stat_info/"+str(setting)+"/"+ind+'_sig.csv')


stat_info()

