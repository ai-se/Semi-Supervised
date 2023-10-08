# -*- coding: utf-8 -*-

"""
Indicators to evaluate the effort-aware defect prediction models
author: Wenzhou Zhang
date: 2018/9/24
"""

import numpy as np


def acc(sortval, loc, indpval):
    """
    :param sortval: sort key (e.g., the prediction value)
    :param loc: x-axis (e.g., LOC_TOTAL, effort)
    :param indpval: y-axis (e.g., bug density)
    :return: acc score
    """
    sortval = np.ravel(sortval)
    loc = np.ravel(loc)
    indpval = np.ravel(indpval)
    bug_n = np.sum(indpval)
    if bug_n == 0:
        return 1
    pel = np.stack([sortval, loc, indpval], axis=0)
    pel = pel[:, np.argsort(pel[1, :])]
    pel = pel[:, np.argsort(-pel[0, :], kind='mergesort')]
    pel = pel[:, pel[1, :].cumsum() <= 0.2 * np.sum(loc)]
    return np.sum(pel[2, :])/bug_n


def calcAUC(cumLOC, cumIndp, norm=True):
    """
    :param cumLOC: x-axis
    :param cumIndp: y-axis
    :param norm: is normalized
    :return: auc
    """
    mat = np.stack([cumLOC, cumIndp], axis=0)
    tmp = mat[:, np.argsort(-mat[1, :])]
    res = tmp[:, np.argsort(tmp[0, :], kind='mergesort')]
    cumIndp = res[1, :]

    # calc diff
    diffLOC = np.diff(cumLOC)
    diffIndp = np.diff(cumIndp)

    # calc AUC of the plot
    auc = (diffLOC * cumIndp[1:len(cumIndp)]) - (diffLOC * diffIndp / 2)

    auc = np.sum(auc) + (cumLOC[0] * cumIndp[0] / 2)
    if norm:
        auc = auc / (cumLOC[-1] * cumIndp[-1])

    return auc


def P_opt(sortval, loc, indpval, dc=False):
    """
    P_opt = 1 - delta_opt
    :param sortval: sort key (e.g., the prediction value)
    :param loc: x-axis (e.g., LOC_TOTAL, effort)
    :param indpval: y-axis (e.g., bug density)
    :param dc: is decreasing
    :return:
    """
    sortval = np.ravel(sortval)
    loc = np.ravel(loc)
    indpval = np.ravel(indpval)

    sign = 1 if dc else -1
    sortedId = np.argsort(sign * sortval)

    # cumulative summing
    cloc = loc[sortedId].cumsum()
    cindp = indpval[sortedId].cumsum()

    # calc optimal model
    optId = np.argsort(-(indpval / loc))
    optcloc = loc[optId].cumsum()
    optcindp = indpval[optId].cumsum()

    minId = np.argsort(indpval / loc)
    mincloc = loc[minId].cumsum()
    mincindp = indpval[minId].cumsum()

    # calc AUC of the plot
    auc = calcAUC(cloc, cindp)
    optauc = calcAUC(optcloc, optcindp)
    minauc = calcAUC(mincloc, mincindp)

    return 1 - ((optauc - auc) / (optauc - minauc))