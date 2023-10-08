import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import random
import numpy as np
import pickle as pkl
from operator import add 
from scipy import stats
from scipy.special import logsumexp
from collections import Counter
from matplotlib import pyplot as plt
import copy
import statistics

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.svm import SVC
from scipy import sparse
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import SpectralClustering,KMeans
from sklearn.linear_model import LinearRegression, Lasso

import SMOTE
from EATT.eatt import EATT
from semisup_learn.methods.qns3vm_old import QN_S3VM
# from safeu.classification.TSVM import TSVM
from mvlearn.semi_supervised import CTClassifier
import metrices as nmetrics

import matlab.engine as engi
import matlab as mat
import math

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df


def prepare_data_commit_guru_file(project):
    file_data_path = '../all_data/defect_prediction/700/commit_guru_file/' + project + '.csv'
    commit_data_path = '../all_data/defect_prediction/700/commit_guru/' + project + '.csv'
    
    commit_data_df = pd.read_csv(commit_data_path)
    commit_data_df = commit_data_df[['commit_hash','contains_bug']]
    commit_data_df['contains_bug'].fillna(False, inplace = True)
    commit_data_df["contains_bug"] = commit_data_df["contains_bug"].astype(int)
    
    data_df = pd.read_csv(file_data_path)
    data_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)
    data_df = data_df.merge(commit_data_df, on = 'commit_hash')
    

    for col in ['id', 'commit_hash', 'file_name']:
        if col in data_df.columns:
            data_df = data_df.drop([col], axis = 1)
            
    data_df = data_df.dropna()
    data_df.reset_index(drop= True, inplace = True)

    y = data_df.contains_bug
    X = data_df.drop(['contains_bug'],axis = 1)
    cols = X.columns
    scaler = MinMaxScaler()
    LOC = X['file_la'] + X['file_lt'] - X['file_ld']
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns = cols)
    data_df = X
    data_df['Bugs'] = y
    data_df['LOC'] = LOC.values.tolist()
    return data_df


def create_model(model):
    supervised_model_list = {'LR': LogisticRegression(), 
                            'DT': DecisionTreeClassifier(), 
                            'RF': RandomForestClassifier(), 
                            'GNB': GaussianNB(), 
                            'SVM': SVC(probability=True)}
    return supervised_model_list[model]



'''
Supervised Model
'''
def supervised_models(model, X_train, y_train):
    clf = model
    clf.fit(X_train, y_train)
    return clf


'''
Self Training
'''
def self_training(clf, X_train, y_train):
    self_training_model = SelfTrainingClassifier(clf)
    self_training_model.fit(X_train, y_train)
    return self_training_model

'''
Label Propagation
'''
def label_propagation(X_train, y_train):
    label_prop_model = LabelPropagation(max_iter=10000)
    label_prop_model.fit(X_train, y_train)
    return label_prop_model

'''
Label Spreading
'''
def label_spreading(X_train, y_train):
    label_spread_model = LabelSpreading(max_iter=10000)
    label_spread_model.fit(X_train, y_train)
    return label_spread_model


'''
Semi Supervised GMM
'''
def semi_GMM(X_train,y_train,X_train_labeled,y_train_labeled):
    gm = GaussianMixture(n_components=2, random_state=0).fit(X_train)
    train_predict = gm.predict(X_train_labeled)
    x_train_df = copy.deepcopy(X_train_labeled)
    x_train_df['y_predict'] = train_predict
    x_train_df['y_actual'] = y_train_labeled

    actual_defect = x_train_df[x_train_df['y_actual'] == 1]
    defect_labe_percentagel = actual_defect[actual_defect['y_predict'] == 1].shape[0]/actual_defect.shape[0]

    if defect_labe_percentagel > 0.5:
        label = {1:1,0:0}
    else:
        label = {1:0,0:1}
    return gm, label

def semi_GMM_predict(gm, label, X_test):
    predicted = gm.predict(X_test)
    updated_predicted = []

    for y_hat in predicted:
        if y_hat == 1:
            updated_predicted.append(label[1])
        else:
            updated_predicted.append(label[0])
    return updated_predicted


'''
Co-Training
'''
def cotraining_single_view(X_train,y_train, estimator1, estimator2):
    l_train = []
    for i in y_train.index:
        y = y_train.loc[i]
        if y == -1:
            y_train.loc[i] = np.nan
    ctc = CTClassifier(estimator1, estimator2, random_state=23)
    ctc = ctc.fit([X_train,X_train], y_train)
    return ctc

def cotraining_multi_view(X_train,y_train, estimator1, estimator2):
    l_train = []
    for i in y_train.index:
        y = y_train.loc[i]
        if y == -1:
            y_train.loc[i] = np.nan

    ctc = CTClassifier(estimator1, estimator2, random_state=23)

    view_1 = random.sample(X_train.columns.tolist(), 15)
    view_2 = random.sample(X_train.columns.tolist(), 15)
    
    
    
    ctc = ctc.fit([X_train[view_1], X_train[view_2]], y_train)
    return ctc, view_1, view_2



'''
Effort-Aware tri-Training
'''
def tri_training(X_train, y_train, effort):
    tt =EATT()
    tt.fit(X_train, y_train, effort)
    return tt

'''
FTcF.MDS
'''
def get_best_d(labeled_df):
    labeled_y = labeled_df.Bugs
    labeled_X = labeled_df.drop(['Bugs'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(labeled_X, labeled_y, test_size=0.33, random_state=42)
    scores = {}
    for i in range(1, labeled_df.shape[1]-1):
        embedding = MDS(n_components=i)
        MDS_X = embedding.fit_transform(X_train)
        clf = RandomForestClassifier()
        clf.fit(MDS_X, y_train)
        MDS_X_test = embedding.fit_transform(X_test)
        predicted = clf.predict(MDS_X_test)
        degree_of_freedom = np.cov(y_test,predicted)[0][1]/statistics.variance(labeled_y)
        GCV = sum((y_test - predicted))**2/(1-degree_of_freedom)**2
        scores[i] = GCV
    return scores

def FTcF_MDS(labeled_df, unlabeled_df, d):
    updated_labeled_df = copy.deepcopy(labeled_df)
    temp_df = copy.deepcopy(unlabeled_df)


    updated_labeled_df.reset_index(drop = True, inplace = True)
    temp_df.reset_index(drop = True, inplace = True)

    embedding = MDS(n_components=d)

    updated_labeled_y = updated_labeled_df.Bugs
    updated_labeled_X = updated_labeled_df.drop(['Bugs'], axis = 1)
    updated_labeled_X = embedding.fit_transform(updated_labeled_X)

    updated_labeled_df = pd.DataFrame(updated_labeled_X)
    updated_labeled_df['Bugs'] = updated_labeled_y

    temp_df_y = temp_df.Bugs
    temp_df_X = temp_df.drop(['Bugs'], axis = 1)
    temp_df_X = embedding.fit_transform(temp_df_X)
    temp_df = pd.DataFrame(temp_df_X)
    temp_df['Bugs'] = temp_df_y

    num_try = 0
    while(temp_df.shape[0] > 0):
        train_y = updated_labeled_df.Bugs
        train_X = updated_labeled_df.drop(['Bugs'], axis = 1)

        unlabeled_X = temp_df.drop(['Bugs'], axis = 1)

        clf = RandomForestClassifier()
        clf.fit(train_X, train_y)
        predicted_prob = clf.predict_proba(unlabeled_X)
        predicted = clf.predict(unlabeled_X)

        PCE = [max(prob) for prob in predicted_prob]

        unlabeled_X['PCE'] = PCE
        unlabeled_X['Bugs'] = predicted

        unlabeled_X = unlabeled_X.sort_values('PCE',ascending=False)

        pseudo_labeled_df = unlabeled_X[unlabeled_X['PCE'] >= 0.9]
        pseudo_labeled_df = pseudo_labeled_df.drop(['PCE'], axis = 1)

        updated_labeled_df = pd.concat([updated_labeled_df,pseudo_labeled_df])

        temp_df = unlabeled_X[unlabeled_X['PCE'] < 0.90]
        temp_df = temp_df.drop(['PCE'], axis = 1)

        num_try += 1

        if num_try  >= 30:
            break
    return clf, embedding

'''
Co Forest
'''
def resampleWithWeights(data):
    data = data.sample(frac = 1)
    data.reset_index(inplace = True, drop = True)
    weights = [1]*data.shape[0]
    for i in range(data.shape[0]):
        weights[i] = data.loc[i,'weight']
    probabilities = [0]*data.shape[0]
    sumProbs = 0
    sumOfWeights = sum(weights)
    for i in range(data.shape[0]):
        sumProbs += round(random.random(),2)
        probabilities[i] = sumProbs

    newData = pd.DataFrame([], columns = data.columns)
    l = 0
    k = 0
    sumProbs = 0
    while((k < data.shape[0]) and (l < data.shape[0])):
        if (weights[l] < 0):
            print('Error:')
        sumProbs += weights[l]
        while ((k < data.shape[0]) and (probabilities[k] <= sumProbs)):
            newData = newData.append(data.iloc[l])
            newData.loc[l,'weight'] = 1
            k += 1
        l += 1
    newData = newData.drop_duplicates()
    return newData

def measureError(clf, data, i, m_threshold):
    sub_df_sum, weights, test_y = get_confidence(clf, data, i)
    confidence = sub_df_sum#[(sub_df_sum <= (1-m_threshold)) | (sub_df_sum >= m_threshold)]
    selected_instances = confidence.index
    count = weights[selected_instances].sum()
    predicted = pd.Series(round(confidence))
    
    error_df = pd.DataFrame([], columns = ['predicted','actual','weights'])
    error_df['predicted'] = predicted
    error_df['actual'] = test_y[selected_instances]
    error_df['weights'] = weights[selected_instances]
    error_df = error_df[error_df['predicted'] != error_df['actual']]
    
    err = error_df['weights'].sum()
    
    print("Error",err/count)
    
    return err/count

def get_confidence(clf, data, i):
    _clf = clf.estimators_[i]
    test_y = data.Bugs
    weights = data.weight
    test_X = data.drop(['Bugs','weight'], axis = 1)
    all_prediction = []
    for _clf in clf.estimators_:
        all_prediction.append(_clf.predict(test_X))
    all_prediction_df = pd.DataFrame(all_prediction)
    sub_df = all_prediction_df.drop(i, axis = 0)
    sub_df_sum = sub_df.sum()/sub_df.shape[0]
    return sub_df_sum, weights, test_y

def coforest(labeled_df, unlabeled_df, val_df):
    labeled_data = copy.deepcopy(labeled_df)
    unlabeled_data = copy.deepcopy(unlabeled_df)
    val_data = copy.deepcopy(val_df)

    labeled_data['weight'] = [1]*labeled_data.shape[0]
    unlabeled_data['weight'] = [0.5]*unlabeled_data.shape[0]
    val_data['weight'] = [1]*val_data.shape[0]

    labeled_data.reset_index(drop = True, inplace = True)
    unlabeled_data.reset_index(drop = True, inplace = True)
    val_data.reset_index(drop = True, inplace = True)

    n_classifier = 30
    m_threshold = 0.75
    err = [0]*n_classifier
    err_prime = [0.5]*n_classifier
    s_prime = [0]*n_classifier
    labeleds = [0]*n_classifier

    clf = RandomForestClassifier(n_estimators = n_classifier, max_features = 'log2')
    train_y = labeled_data.Bugs
    weights = labeled_data.weight
    train_X = labeled_data.drop(['Bugs','weight'], axis = 1)
    clf.fit(train_X,train_y)


    for i in range(n_classifier):
        labeleds[i] = resampleWithWeights(labeled_data)
        _clf = clf.estimators_[i]
        new_train_y = labeleds[i].Bugs
        new_train_X = labeleds[i].drop(['Bugs','weight'], axis = 1)
        _clf.fit(new_train_X,new_train_y)


    unlabeled_X = unlabeled_data.drop(['Bugs', 'weight'], axis = 1)

    probs = clf.predict_proba(unlabeled_X)
    confidence = []
    for prob in probs:
        if prob[1] < 0.5:
            confidence.append(1-prob[1])
        else:
            confidence.append(prob[1])
    unlabeled_data['weight'] = confidence

    bChanged = True
    Li = [pd.DataFrame()]*n_classifier

    while(bChanged):
        bChanged = False
        bUpdate = [False]*n_classifier
        m_classifiers = []*n_classifier
        Li = [pd.DataFrame()]*n_classifier

        for i in range(n_classifier):
            err[i] = measureError(clf, val_data, i, m_threshold)

            if(err[i] <= err_prime[i]):
                if(s_prime[i] == 0):
                    s_prime[i] = min(unlabeled_data.weight.sum() / 10, 100)
                weight = 0
                unlabeled_data = unlabeled_data.sample(frac = 1)
                numWeightsAfterSubsample = round(((err_prime[i] * s_prime[i]) / (err[i]+0.0001) - 1))
                temp_df = pd.DataFrame()
                for k in range(unlabeled_data.shape[0]):
                    weight += unlabeled_data.loc[k,'weight']
                    if (weight > numWeightsAfterSubsample):
                        break
                    temp_df = pd.concat([temp_df,unlabeled_data.iloc[k]], axis = 1)
                temp_df = temp_df.T
                sub_df_sum,_,_ = get_confidence(clf, temp_df, i)
                confidence = sub_df_sum[(sub_df_sum <= (1-m_threshold)) | (sub_df_sum >= m_threshold)]
                predicted = pd.Series(round(confidence))
                predicted.reset_index(drop = True, inplace = True)
                selected_instances = confidence.index
                temp_df = temp_df.iloc[selected_instances]
                temp_df.reset_index(drop = True, inplace = True)
                temp_df['Bugs'] = predicted
                Li[i] = pd.concat([Li[i],temp_df], axis = 0)
                Li[i].drop_duplicates(inplace = True)
                if s_prime[i] < Li[i].shape[0]:
                    if (err[i] * Li[i].weight.sum()) < (err_prime[i] * s_prime[i]):
                        bUpdate[i] = True

        for i in range(n_classifier):
            if bUpdate[i] == True:
                size = Li[i].weight.sum() #min(Li[i].weight.sum()/10, 100)
                bChanged = True
                _clf = clf.estimators_[i]
                labeled_train_y = labeleds[i].Bugs
                labeled_weights = labeleds[i].weight
                labeled_train_X = labeleds[i].drop(['Bugs','weight'], axis = 1)

                unlabeled_train_y = Li[i].Bugs
                unlabeled_weights = Li[i].weight
                unlabeled_train_X = Li[i].drop(['Bugs','weight'], axis = 1)

                pseudo_train_X = pd.concat([labeled_train_X, unlabeled_train_X], axis = 0)
                pseudo_train_y = pd.concat([labeled_train_y, unlabeled_train_y], axis = 0)
                pseudo_weights = pd.concat([labeled_weights, unlabeled_weights], axis = 0)

                _clf.fit(train_X, train_y,  sample_weight = weights)

                err_prime[i] = err[i]
                s_prime[i] = size
    return clf

'''
Semi Booste
'''
def semiBooste(X_train_mixedlabeled, y_train_mixedlabeled, clf):
    boost_clf = SemiBoostClassifier(base_model = clf)
    _X = X_train_mixedlabeled.reset_index(drop = True, inplace = False)
    _y = y_train_mixedlabeled.reset_index(drop = True, inplace = False)
    boost_clf.fit(_X, _y)
    return boost_clf

class SemiBoostClassifier():

    def __init__(self, base_model = SVC()):

        self.BaseModel = base_model

    def fit(self, X, y,
            n_neighbors=4, n_jobs = 1,
            max_models = 25,
            sample_percent = 0.5,
            sigma_percentile = 90,
            labels = [1.0,0.0],
            similarity_kernel = 'rbf',
            verbose = True):

        ''' Fit model'''
        # Localize labeled data
        idx_label = np.array(y[y != -1.0].index)
        idx_not_label = np.array(y[y == -1.0].index)
        

        # The parameter C is defined in the paper as
        C = idx_label.shape[0]/idx_not_label.shape[0]

        # First we need to create the similarity matrix
        if similarity_kernel == 'knn':

            self.S = neighbors.kneighbors_graph(X,
                                                n_neighbors=n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=n_jobs)

            self.S = sparse.csr_matrix(self.S)

        elif similarity_kernel == 'rbf':
            # First aprox
            self.S = np.sqrt(rbf_kernel(X, gamma = 1))
            # set gamma parameter as the 15th percentile
            sigma = np.percentile(np.log(self.S), sigma_percentile)
            sigma_2 = (1/sigma**2)*np.ones((self.S.shape[0],self.S.shape[0]))
            self.S = np.power(self.S, sigma_2)
            # Matrix to sparse
            self.S = sparse.csr_matrix(self.S)
        else:
            print('No kernel type ', similarity_kernel)

        self.models = []
        self.weights = []
        H = np.zeros(idx_not_label.shape[0])

        # Loop for adding sequential models
        for t in range(max_models):
            try:
                p_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==1.0))[idx_not_label]*np.exp(-2*H)
                p_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(H))[idx_not_label]*np.exp(-H)
                p = np.add(p_1, p_2)
                p = np.squeeze(np.asarray(p))

                q_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==0.0))[idx_not_label]*np.exp(2*H)
                q_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(-H))[idx_not_label]*np.exp(H)
                q = np.add(q_1, q_2)
                q = np.squeeze(np.asarray(q))

                z = np.sign(p-q)
                z[z==-1.0] = 0
                z_conf = np.abs(p-q)

                sample_weights = z_conf/np.sum(z_conf)
                if np.any(sample_weights != 0):
                    idx_aux = np.random.choice(np.arange(len(z)),
                                                  size = int(sample_percent*len(idx_not_label)),
                                                  p = sample_weights,
                                                  replace = False)
                    idx_sample = idx_not_label[idx_aux]

                else:
                    print('No similar unlabeled observations left.')
                    break

                idx_total_sample = np.concatenate([idx_label,idx_sample])
                X_t = X.loc[idx_total_sample,]

                y.loc[idx_sample] = z[idx_aux]
                y_t = y.loc[idx_total_sample]

                clf = self.BaseModel
                clf.fit(X_t, y_t)

                h = clf.predict(X.loc[idx_not_label])

                idx_label = idx_total_sample
                idx_not_label = np.array(y[y == -1.0].index)

                if verbose:
                    print('There are still ', idx_not_label.shape[0], ' unlabeled observations')

                e = (np.dot(p,h==-1) + np.dot(q,h==1))/(np.sum(np.add(p,q)))
                a = 0.25*np.log((1-e)/e)

                if a<0:
                    if verbose:
                        print('Problematic convergence of the model. a<0')
                    break

                self.models.append(clf)
                self.weights.append(a)

                H = np.zeros(len(idx_not_label))
                w = np.sum(self.weights)
                for i in range(len(self.models)):
                    H = np.add(H, self.weights[i]*self.models[i].predict(X.loc[idx_not_label]))
                if (t==max_models) & verbose:
                    print('Maximum number of models reached')

                if len(idx_not_label) == 0:
                    if verbose:
                        print('All observations have been labeled')
                        print('Number of iterations: ',t + 1)
                    break
            except:
                break

        if verbose:
            print('\n The model weights are \n')
            print(self.weights)



    def predict(self, X):
        estimate = np.zeros(X.shape[0])
        w = np.sum(self.weights)
        for i in range(len(self.models)):
            estimate = np.add(estimate, self.weights[i]*self.models[i].predict(X))
        estimate = np.array(list(map(lambda x: 1.0 if x>0 else 0.0, estimate)))
        estimate = estimate.astype(int)
        return estimate

'''
S3VM
'''
def S3VM(X_train_mixedlabeled, y_train_mixedlabeled):
    clf = SKTSVM()
    _X = copy.deepcopy(X_train_mixedlabeled)
    _y = copy.deepcopy(y_train_mixedlabeled)
    _X.reset_index(inplace=True, drop = True)
    _y.reset_index(inplace=True, drop = True)
    clf.fit(_X, _y)
    return clf
class SKTSVM(BaseEstimator):
    """
    Scikit-learn wrapper for transductive SVM (SKTSVM)
    
    Wraps QN-S3VM by Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer (see http://www.fabiangieseke.de/index.php/code/qns3vm) 
    as a scikit-learn BaseEstimator, and provides probability estimates using Platt scaling

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be 'linear' or 'rbf'

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf'

    lamU: float, optional (default=1.0) 
        cost parameter that determines influence of unlabeled patterns
        must be float >0

    probability: boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    """
    
    # lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
    def __init__(self, kernel = 'RBF', C = 1e-4, gamma = 0.5, lamU = 1.0, probability=True):
        self.random_generator = random.Random()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma 
        self.lamU = lamU
        self.probability = probability
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X
            Must be 0 or 1 for labeled and -1 for unlabeled instances 

        Returns
        -------
        self : object
            Returns self.
        """
        
        # http://www.fabiangieseke.de/index.php/code/qns3vm
        
        unlabeled_idx = y[y == -1.0].index
        labeled_idx = y[y != -1.0].index
        
        unlabeledX = X.loc[unlabeled_idx].values.tolist()
        labeledX = X.loc[labeled_idx].values.tolist()
        labeledy = y.loc[labeled_idx]
        
        # convert class 0 to -1 for tsvm
        labeledy[labeledy == 0] = -1
        labeledy = labeledy.tolist()
        
        if 'rbf' in self.kernel.lower():
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.C, lamU=self.lamU, kernel_type="RBF", sigma=self.gamma)
        else:
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.C, lamU=self.lamU)
            
        self.model.train()
        
        # probabilities by Platt scaling
        if self.probability:
            self.plattlr = LR()
            preds = self.model.mygetPreds(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
        
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if self.probability:
            preds = self.model.mygetPreds(X.tolist())
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))
        else:
            raise RuntimeError("Probabilities were not calculated for this model - make sure you pass probability=True to the constructor")
        
    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        y = np.array(self.model.getPredictions(X.values.tolist()))
        y[y == -1] = 0
        return y
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

'''
LapSVMp
'''
def lapSvmp(X_train_mixedlabeled, y_train_mixedlabeled, X_test,y_test):
    eng = engi.start_matlab()
    eng.addpath(r'lapsvmp_v02/',nargout=0)
    X_train_mixedlabeled = pd.concat([X_train_mixedlabeled,X_test], axis = 0)
    y_train_mixedlabeled = pd.concat([y_train_mixedlabeled,y_test], axis = 0)
    y_train_mixedlabeled = y_train_mixedlabeled.replace({0:-1, -1:0})

    X_train_mixedlabeled.reset_index(inplace=True, drop=True)
    y_train_mixedlabeled.reset_index(inplace=True, drop=True)
    
    mat_X = mat.double(X_train_mixedlabeled.values.tolist())
    mat_y = mat.double(y_train_mixedlabeled.values.tolist())
    k = eng.run_code(mat_X, mat_y, nargout=1) 
    k_a = np.asarray(k)
    k_a.flatten()
    predicted = pd.Series(k_a.flatten())
    predicted = predicted.astype(int)
    predicted = predicted.replace({-1:0})
    predicted = predicted.tail(X_test.shape[0])
    return predicted


'''
get_data
'''
def get_data(X_train, X_test, y_train, y_test, loc_data):
    unlabeled_y = []

    y_train_supervised = copy.deepcopy(y_train)
    X_train_supervised = copy.deepcopy(X_train)

    for y in y_train:
        if np.random.binomial(1, p=0.67) == 1:
            unlabeled_y.append(-1)
        else:
            unlabeled_y.append(y)

    mixed_train_df = copy.deepcopy(X_train)
    mixed_train_df['Bugs'] = unlabeled_y
    labeled_df = mixed_train_df[mixed_train_df['Bugs'] != -1]
    unlabeled_df = mixed_train_df[mixed_train_df['Bugs'] == -1]

    y_train_unlabeled = unlabeled_df.Bugs
    X_train_unlabeled = unlabeled_df.drop(['Bugs'], axis = 1)

    y_train_labeled = labeled_df.Bugs
    X_train_labeled = labeled_df.drop(['Bugs'], axis = 1)

    X_train, X_val, y_train, y_val = train_test_split(X_train_labeled, y_train_labeled, test_size=0.33, random_state=42)

    val_df = copy.deepcopy(X_val)
    val_df['Bugs'] = y_val

    X_train_labeled = copy.deepcopy(X_train)
    y_train_labeled = copy.deepcopy(y_train)

    labeled_df = copy.deepcopy(X_train_labeled)
    labeled_df['Bugs'] = y_train_labeled

    labeled_df = apply_smote(labeled_df)
    
    labeled_df['Bugs'] = labeled_df.Bugs.astype(int)

    y_train_labeled = labeled_df.Bugs
    X_train_labeled = labeled_df.drop(['Bugs'], axis = 1)

    X_train_mixedlabeled = pd.concat([X_train_labeled,X_train_unlabeled], axis = 0)
    y_train_mixedlabeled = pd.concat([y_train_labeled,y_train_unlabeled], axis = 0)

    X_train_mixedlabeled.reset_index(inplace=True, drop = True)
    y_train_mixedlabeled.reset_index(inplace=True, drop = True)

    effort = ((X_train_mixedlabeled['file_ld']+X_train_mixedlabeled['file_la'])*X_train_mixedlabeled['file_lt']*1).values/2 + 1
    return [X_train_mixedlabeled, y_train_mixedlabeled, X_train_labeled, y_train_labeled, X_test, y_test, X_val, y_val, labeled_df, unlabeled_df, effort, loc_data, X_train_supervised, y_train_supervised]


'''
run classifier
'''

def compile_results(y_test, predicted, loc, results):
    abcd = nmetrics.measures(y_test,predicted,loc)
    results['f1'].append(abcd.calculate_f1_score())
    results['precision'].append(abcd.calculate_precision())
    results['recall'].append(abcd.calculate_recall())
    results['g-score'].append(abcd.get_g_score())
    results['d2h'].append(abcd.calculate_d2h())
    results['pci_20'].append(abcd.get_pci_20())
    results['ifa'].append(abcd.get_ifa())
    results['pd'].append(abcd.get_pd())
    results['pf'].append(abcd.get_pf())
    return results
    
    
def initial_results():
    results = {
        'f1':[], 
        'precision':[], 
        'recall':[], 
        'g-score':[], 
        'd2h':[], 
        'pci_20':[],
        'ifa':[], 
        'pd':[],
        'pf':[]
        }
    return results


def run_classifiers(project, supervised_model_list):

    skf = StratifiedKFold(n_splits=3, random_state=None)

    df = prepare_data_commit_guru_file(project)
    y = df.Bugs
    loc_data = df.LOC
    X = df.drop(['Bugs', 'LOC'], axis = 1)
    y = y.astype(int)

    all_results = {}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        data = get_data(X_train, X_test, y_train, y_test, loc_data)

        X_train_mixedlabeled = data[0]
        y_train_mixedlabeled = data[1]
        X_train_labeled = data[2]
        y_train_labeled = data[3]
        X_test = data[4]
        y_test = data[5]
        X_val = data[6]
        y_val = data[7]
        labeled_df = data[8]
        unlabeled_df = data[9]
        effort = data[10]
        loc_data  = data[11]
        X_train_supervised = data[12]
        y_train_supervised = data[13]
        loc = loc_data[X_test.index]

        for model in supervised_model_list.keys():
            if model not in all_results.keys():
                all_results[model] = copy.deepcopy(initial_results())
            clf = supervised_models(create_model(model), X_train_supervised, y_train_supervised)
            predicted = clf.predict(X_test)
            all_results[model] = compile_results(y_test, predicted, loc, all_results[model])

        print("Supervised Models Done")

        for model in supervised_model_list.keys():
            self_training_model = 'self_training_' + model
            if self_training_model not in all_results.keys():
                all_results[self_training_model] = copy.deepcopy(initial_results())
            clf = create_model(model)
            _X = copy.deepcopy(X_train_mixedlabeled)
            _y = copy.deepcopy(y_train_mixedlabeled)
            clf = self_training(clf, _X, _y)
            predicted = clf.predict(X_test)
            all_results[self_training_model] = compile_results(y_test, predicted, loc, all_results[self_training_model])
        
        print("Self Training Models Done")
        
        if 'LabelPropagation' not in all_results.keys():
            all_results['LabelPropagation'] = copy.deepcopy(initial_results())
        _X = copy.deepcopy(X_train_mixedlabeled)
        _y = copy.deepcopy(y_train_mixedlabeled)
        label_prop_model = label_propagation(_X, _y)
        predicted = label_prop_model.predict(X_test)
        all_results['LabelPropagation'] = compile_results(y_test, predicted, loc, all_results['LabelPropagation'])

        if 'LabelSpreading' not in all_results.keys():
            all_results['LabelSpreading'] = copy.deepcopy(initial_results())
        _X = copy.deepcopy(X_train_mixedlabeled)
        _y = copy.deepcopy(y_train_mixedlabeled)
        label_spread_model = label_spreading(_X, _y)
        predicted = label_spread_model.predict(X_test)
        all_results['LabelSpreading'] = compile_results(y_test, predicted, loc, all_results['LabelSpreading'])

        print("Graph Training Models Done")

        if 'Semi_GMM' not in all_results.keys():
            all_results['Semi_GMM'] = copy.deepcopy(initial_results())
        _X = copy.deepcopy(X_train_mixedlabeled)
        _y = copy.deepcopy(y_train_mixedlabeled)
        _Xl = copy.deepcopy(X_train_labeled)
        _yl = copy.deepcopy(y_train_labeled)
        gm, label = semi_GMM(_X,_y,_Xl,_yl)
        _Xt = copy.deepcopy(X_test)
        predicted = semi_GMM_predict(gm, label, _Xt)
        all_results['Semi_GMM'] = compile_results(y_test, predicted, loc, all_results['Semi_GMM'])

        print("Cluster Models Done")

        for model_1 in supervised_model_list.keys():
            for model_2 in supervised_model_list.keys():
                co_training_sv_model = 'co_training_sv_' + model_1 + '_' + model_2
                if co_training_sv_model not in all_results.keys():
                    all_results[co_training_sv_model] = copy.deepcopy(initial_results())
                estimator1 = create_model(model_1)
                estimator2 = create_model(model_2)
                _X = copy.deepcopy(X_train_mixedlabeled)
                _y = copy.deepcopy(y_train_mixedlabeled)
                ctc = cotraining_single_view(_X, _y, estimator1, estimator2)
                predicted = ctc.predict([X_test,X_test])
                all_results[co_training_sv_model] = compile_results(y_test, predicted, loc, all_results[co_training_sv_model])

        print("co_training_sv_model Models Done")

        for model_1 in supervised_model_list.keys():
            for model_2 in supervised_model_list.keys():
                co_training_mv_model = 'co_training_mv_' + model_1 + '_' + model_2
                if co_training_mv_model not in all_results.keys():
                    all_results[co_training_mv_model] = copy.deepcopy(initial_results())
                estimator1 = create_model(model_1)
                estimator2 = create_model(model_2)
                _X = copy.deepcopy(X_train_mixedlabeled)
                _y = copy.deepcopy(y_train_mixedlabeled)
                ctc, view_1, view_2 = cotraining_multi_view(_X, _y, estimator1, estimator2)
                predicted = ctc.predict([X_test[view_1], X_test[view_2]])
                all_results[co_training_mv_model] = compile_results(y_test, predicted, loc, all_results[co_training_mv_model])

        print("co_training_mv_model Models Done")

        if 'EATT' not in all_results.keys():
            all_results['EATT'] = copy.deepcopy(initial_results())
        clf = tri_training(X_train_mixedlabeled,y_train_mixedlabeled,effort)
        predicted = clf.predict(X_test)
        all_results['EATT'] = compile_results(y_test, predicted, loc, all_results['EATT'])

        print("EATT Models Done")

        if 'FTcF_MDS' not in all_results.keys():
            all_results['FTcF_MDS'] = copy.deepcopy(initial_results())
            scores = get_best_d(labeled_df)
            d = max(scores, key= lambda x: scores[x])  
        clf, embedding = FTcF_MDS(labeled_df, unlabeled_df, d)
        X_test_MDS = embedding.fit_transform(X_test)
        predicted = clf.predict(X_test_MDS)
        all_results['FTcF_MDS'] = compile_results(y_test, predicted, loc, all_results['FTcF_MDS'])

        if 'coForest' not in all_results.keys():
            all_results['coForest'] = copy.deepcopy(initial_results())
        val_df = copy.deepcopy(X_val)
        val_df['Bugs'] = y_val
        clf = coforest(labeled_df, unlabeled_df, val_df)
        predicted = clf.predict(X_test)
        all_results['coForest'] = compile_results(y_test, predicted, loc, all_results['coForest'])

        for model in supervised_model_list.keys():
            boosting_model = 'boosting_' + model
            if boosting_model not in all_results.keys():
                all_results[boosting_model] = copy.deepcopy(initial_results())
            base_clf = create_model(model)
            clf = semiBooste(X_train_mixedlabeled, y_train_mixedlabeled, base_clf)
            predicted = clf.predict(X_test)
            all_results[boosting_model] = compile_results(y_test, predicted, loc, all_results[boosting_model])


        if 'S3VM' not in all_results.keys():
            all_results['S3VM'] = copy.deepcopy(initial_results())
        clf = S3VM(X_train_mixedlabeled, y_train_mixedlabeled)
        predicted = clf.predict(X_test)
        all_results['S3VM'] = compile_results(y_test, predicted, loc, all_results['S3VM'])

        if 'lapSvm' not in all_results.keys():
            all_results['lapSvm'] = copy.deepcopy(initial_results())
        predicted = lapSvmp(X_train_mixedlabeled, y_train_mixedlabeled, X_test,y_test)
        all_results['lapSvm'] = compile_results(y_test, predicted, loc, all_results['lapSvm'])

    result_path = 'results/Project_specific_lap/' + project + '.pkl'
    with open(result_path, 'wb') as handle:
        pkl.dump(all_results, handle, protocol=pkl.HIGHEST_PROTOCOL)

def run(projects, supervised_model_list):
    for project in projects:
        try:
            print(project)
            run_classifiers(project, supervised_model_list)
        except Exception as e:
            print(e, project)
            continue
    return None



if __name__=="__main__":
    existing_data_source = '../all_data/defect_prediction/700/commit_guru_file/'
    existing_projects = [f.split('.')[0] for f in listdir(existing_data_source) if isfile(join(existing_data_source, f))]

    supervised_model_list = {'LR': LogisticRegression(max_iter=10000), 
                            'DT': DecisionTreeClassifier(), 
                            'RF': RandomForestClassifier(n_estimators = 30), 
                            'GNB': GaussianNB(), 
                            'SVM': SVC(probability=True)}

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    threads = []
    cores = cpu_count()
    existing_projects = existing_projects[start:end]
    split_projects = np.array_split(existing_projects, cores)

    for i in range(cores):
        print("starting thread ",i)
        t = ThreadWithReturnValue(target = run, args = [split_projects[i], supervised_model_list])
        threads.append(t)
    
    for th in threads:
        th.start()
    for th in threads:
        response = th.join()