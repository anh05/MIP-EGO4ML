#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mipego4ml.ConditionalSpace import ConditionalSpace
from mipego4ml.ConfigSpace import ConfigSpace
from mipego4ml.mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
from mipego4ml.mipego4ML import mipego4ML

np.random.seed(123)

dim = 2
n_step = 100
n_init_sample = 15


def read_dataset(filename):
    ilink = './imb_IRlowerThan9/' + filename + '/' + filename + '/' + filename + '.csv'
    glass_data = pd.read_csv(ilink)
    columns = len(glass_data.columns)
    if (filename == "abalone19"):
        X = glass_data.iloc[:, 2:columns - 1]
    else:
        X = glass_data.iloc[:, 1:columns - 1]
    y = glass_data['class']
    X = StandardScaler().fit_transform(X)
    X = np.c_[X]
    return X, y


def adapt_smo1(randomstate):
    alg_namestr = NominalSpace(["SVM", "RF"], "alg_namestr")
    kernel = NominalSpace(["linear", "rbf", "poly", "sigmoid"], "kernel", alg_namestr, "SVM")
    pro_bas = NominalSpace(["true"], 'probability', alg_namestr, "SVM")
    C = ContinuousSpace([1e-2, 100], "C", alg_namestr, "SVM")
    degree = OrdinalSpace([1, 5], 'degree', alg_namestr, "SVM")
    coef0 = ContinuousSpace([0.0, 10.0], 'coef0', alg_namestr, "SVM")
    shrinking = NominalSpace(["true", "false"], "shrinking", alg_namestr, "SVM")
    gamma = NominalSpace(["auto", "value"], "gamma", alg_namestr, "SVM")  # only rbf, poly, sigmoid
    gamma_value = ContinuousSpace([1e-2, 100], "gamma_value", alg_namestr, "SVM")
    # random_state = NominalSpace([int(randomstate)], 'random_state_ML')
    # RF
    RF_n_estimators = OrdinalSpace([5, 2000], "n_estimators", alg_namestr, "RF")
    RF_criterion = NominalSpace(["gini", "entropy"], "criterion", alg_namestr, "RF")
    RF_max_depth = OrdinalSpace([10, 200], "max_depth", alg_namestr, "RF")
    RF_max_features = NominalSpace(['auto', 'sqrt', 'log2'], "max_features", alg_namestr, "RF")
    RF_min_samples_split = ContinuousSpace([0.0, 1.0], "min_samples_split", alg_namestr, "RF")
    RF_min_samples_leaf = ContinuousSpace([0.0, 0.5], "min_samples_leaf", alg_namestr, "RF")
    RF_bootstrap = NominalSpace(["true", "false"], "bootstrap", alg_namestr, "RF")
    RF_class_weight = NominalSpace(['balanced'], "class_weight", alg_namestr, "RF")
    # print(alg_namestr)
    # search_space= ConfigurationSpace()
    # search_space.add_multiparameter([alg_namestr , kernel , pro_bas , C , degree , coef0 , shrinking ,gamma , gamma_value
    #                               , random_state , RF_n_estimators , RF_criterion , RF_max_depth ,RF_max_features ,
    #                              RF_min_samples_split , RF_min_samples_leaf , RF_bootstrap , RF_class_weight])
    search_space = alg_namestr * kernel * pro_bas * C * degree * coef0 * shrinking * gamma * gamma_value * RF_n_estimators * RF_criterion * RF_max_depth * RF_max_features \
                   * RF_min_samples_split * RF_min_samples_leaf * RF_bootstrap * RF_class_weight
    # cs.add_hyperparameters([n_neighbors,weights,algorithm,leaf_size, p, metric])
    # if (alg_namestr == 'SVM'):
    #    search_space = alg_namestr * kernel * pro_bas * C * degree * coef0 * shrinking * gamma * gamma_value * random_state
    # elif(alg_namestr == 'RF'):
    #    search_space = RF_n_estimators *RF_criterion*RF_max_depth *RF_max_features *RF_min_samples_split* RF_min_samples_leaf* RF_min_samples_leaf*RF_bootstrap*RF_class_weight
    return search_space


def adapt_smo(randomstate):
    alg_namestr = NominalSpace(["SVM", "RF"], "alg_namestr")
    # SVM Parameter
    kernel = NominalSpace(["linear", "rbf", "poly", "sigmoid"], "kernel")
    C = ContinuousSpace([1e-2, 100], "C")
    # RandomForest Parameter
    RF_criterion = NominalSpace(["gini", "entropy"], "criterion")
    RF_max_depth = OrdinalSpace([10, 200], "max_depth")
    search_space = ConfigSpace()
    search_space.add_multiparameter([alg_namestr, kernel, RF_criterion, RF_max_depth])
    con = ConditionalSpace("abc")
    con.addMutilConditional([kernel, C], alg_namestr, "SVM")
    con.addMutilConditional([RF_criterion, RF_max_depth], alg_namestr, "RF")

    # print(alg_namestr)

    pro_bas = NominalSpace(["true"], 'probability')
    C = ContinuousSpace([1e-2, 100], "C")
    degree = OrdinalSpace([1, 5], 'degree')
    coef0 = ContinuousSpace([0.0, 10.0], 'coef0')
    shrinking = NominalSpace(["true", "false"], "shrinking")
    gamma = NominalSpace(["auto", "value"], "gamma")  # only rbf, poly, sigmoid
    gamma_value = ContinuousSpace([1e-2, 100], "gamma_value")
    # con= ConditionalSpace("abc")
    con.addMutilConditional([kernel, pro_bas, C, degree, coef0, shrinking, gamma, gamma_value], alg_namestr, "SVM")
    # random_state = NominalSpace([randomstate], 'random_state_ML')
    # RF

    RF_n_estimators = OrdinalSpace([5, 2000], "n_estimators")

    RF_max_features = NominalSpace(['auto', 'sqrt', 'log2'], "max_features")
    RF_min_samples_split = ContinuousSpace([0.0, 1.0], "min_samples_split")
    RF_min_samples_leaf = ContinuousSpace([0.0, 0.5], "min_samples_leaf")
    RF_bootstrap = NominalSpace(["true", "false"], "bootstrap")
    RF_class_weight = NominalSpace(['balanced'], "class_weight")
    con.addMutilConditional([RF_n_estimators, RF_criterion, RF_max_depth, RF_max_features,
                             RF_min_samples_split, RF_min_samples_leaf, RF_bootstrap, RF_class_weight], alg_namestr,
                            "RF")
    # print(alg_namestr)
    search_space = ConfigSpace()
    search_space.add_multiparameter([alg_namestr, kernel, pro_bas, C, degree, coef0, shrinking, gamma, gamma_value
                                        , RF_n_estimators, RF_criterion, RF_max_depth, RF_max_features,
                                     RF_min_samples_split, RF_min_samples_leaf, RF_bootstrap, RF_class_weight])

    # search_space = alg_namestr * kernel * pro_bas * C * degree * coef0 * shrinking * gamma * gamma_value * random_state * RF_n_estimators * RF_criterion * RF_max_depth * RF_max_features * RF_min_samples_split * RF_min_samples_leaf * RF_min_samples_leaf * RF_bootstrap * RF_class_weight
    # cs.add_hyperparameters([n_neighbors,weights,algorithm,leaf_size, p, metric])
    # if (alg_namestr == 'SVM'):
    #    search_space = alg_namestr * kernel * pro_bas * C * degree * coef0 * shrinking * gamma * gamma_value * random_state
    # elif(alg_namestr == 'RF'):
    #    search_space = RF_n_estimators *RF_criterion*RF_max_depth *RF_max_features *RF_min_samples_split* RF_min_samples_leaf* RF_min_samples_leaf*RF_bootstrap*RF_class_weight
    return search_space, con


iid = 0


def next_id():
    global iid
    # print(iid)
    # res = iid
    iid = iid + 1
    # print(iid)
    return iid


def obj_func(params):
    iid = next_id()
    global best, list_bests, list_losts, HPOalg, dataset, smo_name, list_log
    # print(HPOalg)
    print(params)
    # p_probability = True
    parambk = params
    params = {k: params[k] for k in params if params[k]}
    ifError = 0
    abc = time.time() - starttime
    classifier = params['alg_namestr']
    params.pop("alg_namestr", None)
    p_random_state = 27  # params['random_state_ML']
    # params.pop("random_state_ML", None)
    print(params)
    clf = RandomForestClassifier()
    if (classifier == 'SVM'):
        params["probability"] = True if params["probability"] == "true" else False
        p_probability = params['probability']
        params.pop("probability", None)
        p_C = params['C']
        params.pop("C", None)
        p_kernel = params['kernel']
        params.pop("kernel", None)
        if (params['gamma'] == "value"):
            p_gamma = params['gamma_value']
        else:
            p_gamma = params['gamma']
        params.pop("gamma", None)
        params.pop("gamma_value", None)
        params["shrinking"] = True if params["shrinking"] == "true" else False
        p_shrinking = params['shrinking']
        params.pop("shrinking", None)
        p_coef0 = params['coef0']
        p_degree = params['degree']
        params.pop("coef0", None)
        params.pop("degree", None)
        clf = SVC(probability=p_probability, max_iter=299, C=p_C, random_state=p_random_state, kernel=p_kernel,
                  shrinking=p_shrinking, gamma=p_gamma, coef0=p_coef0, degree=p_degree)
    elif (classifier == 'RF'):
        p_n_estimators = params['n_estimators']
        params.pop("n_estimators", None)
        p_criterion = params['criterion']
        params.pop("criterion", None)
        p_max_depth = params['max_depth']
        params.pop("max_depth", None)
        p_max_features = params['max_features']
        params.pop("max_features", None)
        p_min_samples_split = params['min_samples_split']
        params.pop("min_samples_split", None)
        p_min_samples_leaf = params['min_samples_leaf']
        params.pop("min_samples_leaf", None)
        params["bootstrap"] = True if params["bootstrap"] == "true" else False
        p_bootstrap = params['bootstrap']
        params.pop("bootstrap", None)
        p_class_weight = params['class_weight']
        params.pop("class_weight", None)
        # clf = RandomForestClassifier(n_estimators = p_n_estimators, criterion = p_criterion, max_depth = p_max_depth, max_features= p_max_features, min_samples_split = p_min_samples_split, min_samples_leaf = p_min_samples_leaf, bootstrap = p_bootstrap, class_weight = p_class_weight)
        clf = RandomForestClassifier(n_estimators=p_n_estimators, criterion=p_criterion, max_depth=p_max_depth,
                                     max_features=p_max_features, min_samples_split=p_min_samples_split,
                                     min_samples_leaf=p_min_samples_leaf, bootstrap=p_bootstrap,
                                     class_weight=p_class_weight, random_state=p_random_state)
    # print('abc:',p_probability, p_random_state)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # p_sub_type = smo_name
    smo = SMOTETomek(random_state=randomstate)
    for train, test in cv.split(X, y):
        X_smo_train, y_smo_train = smo.fit_sample(X[train], y[train])
        probas_ = clf.fit(X_smo_train, y_smo_train).predict_proba(X[test])
        # print('just [pass1]')
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    auc_loss = 1 - mean_auc
    abc = time.time() - starttime
    print("AUC:", auc_loss)
    if mean_auc > best:
        print('\033[91m', 'NEW BEST:', mean_auc, '\033[0m')
        best = mean_auc
        # print(iid, best, abc, params)
        list_bests.append([HPOalg, iid, dataset, mean_auc, auc_loss, abc, params])
        # list_losts.append([iid,dataset, p_sub_type, auc_loss, abc, params])
    # return {'loss': -acc, 'status': STATUS_OK}
    # col_names = ['current_best','run_time','loss','mean', 'iteration','params','db','smo']
    list_log.append([best, abc, auc_loss, mean_auc, iid, params, dataset])
    return auc_loss


# C = ContinuousSpace([-5, 5],'C') * 2
# I = OrdinalSpace([-100, 100],'I')
# N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], 'N')
list_bests = []
# search_space = C * I * N
dataset = 'glass1'
randomstate = 27
list_log = []
X, y = read_dataset(dataset)
X = StandardScaler().fit_transform(X)
X = np.c_[X]
k = 5
cv = StratifiedKFold(n_splits=k, random_state=randomstate)
search_space, con = adapt_smo(randomstate)
# model = RandomForest(levels=search_space.levels)
# model = RrandomForest(levels=search_space.levels, seed=1, max_features='sqrt')


# print(search_space.levels)
global startime
# print(smo_name, dataset)
starttime = time.time()
ran_best = 0
best = 0
iid = 0

# next we define the surrogate model and the optimizer.
HPOalg = 'H.WANG'
eta = 3
logeta = lambda x: log(x) / log(eta)
opt = mipego4ML(search_space, con, obj_func, minimize=True,  # the problem is a minimization problem.
                max_eval=50,  # we evaluate maximum 500 times
                max_iter=50,  # we have max 500 iterations
                infill='EI',  # Expected improvement as criteria
                n_init_sample=3,  # We start with 10 initial samples
                n_point=1,  # We evaluate every iteration 1 time
                n_job=1,  # with 1 process (job).
                optimizer='MIES',  # We use the MIES internal optimizer.
                verbose=False, random_seed=None)
a, b = opt.run()
print(b)
