from __future__ import print_function
import pdb
from copy import copy
import six
from copy import deepcopy
from collections import defaultdict, deque, OrderedDict
from typing import Union, List, Any, Dict, Iterable, Set, Tuple, Optional
import numpy as np
from numpy.random import randint, rand
from abc import abstractmethod
from pyDOE import lhs
import copy
from itertools import chain
import itertools
import numpy as np
#import mipego4ml.ConditionalSpace
from mipego4ml.ConditionalSpace import ConditionalSpace
# from .mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace
from .mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace, SearchSpace


class ConfigSpace(object):
    def __init__(self, name: Union[str, None] = None,
                 seed: Union[int, None] = None,
                 meta: Optional[Dict] = None,
                 ) -> None:
        self.name = name
        self.meta = meta
        self.random = np.random.RandomState(seed)
        self._hyperparameters = OrderedDict()  # type: OrderedDict[str, Hyperparameter]
        self._hyperparameter_idx = dict()  # type: Dict[str, int]
        self._idx_to_hyperparameter = dict()  # type: Dict[int, str]
        self._listconditional = OrderedDict()
        self._sampler = OrderedDict()
        self._OrgLevels = OrderedDict()
        # self.var_name = np.array_str()
        self.dim = 0
        # self._sampler_updated = OrderedDict()

    def __len__(self):
        return self.dim

    def __iter__(self):
        pass

    def add_multiparameter(self, params: List[SearchSpace]) -> List[SearchSpace]:
        # listParent =OrderedDict()
        for param in params:
            if not isinstance(param, SearchSpace):
                raise TypeError("Hyperparameter '%s' is not an instance of "
                                "mipego.SearchSpace" %
                                str(param))
        for param in params:
            self._add_singleparameter(param)

        return params

    def _add_singleparameter(self, param: SearchSpace) -> None:
        if param.var_name[0] in self._hyperparameters:
            raise ValueError("Hyperparameter '%s' is already in the "
                             "configuration space." % param.var_name[0])
        self._hyperparameters[str(param.var_name[0])] = param
        for i, hp in enumerate(self._hyperparameters):
            if not 'var_name' in locals():
                var_name = np.array([hp])
            else:
                var_name = np.r_[var_name, np.array([hp])]
            self._hyperparameter_idx[hp] = i
        self.var_name = np.array([str(var_name)])

    def _listoutsinglebraches(self, rootnode, rootname, hpi, final, lsvarname, childeffect, lsparentname, con: ConditionalSpace):
        i = rootnode
        hp = deepcopy(self._hyperparameters)
        name = rootname
        final = final
        # print(i)
        if (i[0] in lsvarname):
            #hpa = [(hp1[0], i[1], True) for hp1 in hp if (hp1[0] == i[0])]
            temp= hp[i[0]]

            if(isinstance(i[1], tuple)):
                temp.bounds = [i[1]]
            else:
                temp.bounds = [tuple([i[1]])]
            temp.iskeep = True
            temp.rebuild()
            hpa=[temp]
            child_hpa = [x[1] for x in childeffect if (x[0] == (i[0] + "_" + str(i[1])))]
            # child_node=[x for x in abc if x[0] in child_hpa]
            child_node = []
            # print(len(child_hpa),hpa)
            if (len(child_hpa) > 0):
                if (child_hpa[0] in lsvarname):
                    child_node = [x for x in lsparentname if x[0] in child_hpa]
                else:
                    #child_node = [x[1] for x in con.items() if x[1][0] in child_hpa and x[1][2] == i[1]]
                    child_node = [x for x in con.conditional.values() if x[0] in child_hpa and x[2] == i[1]]
            else:
                hpi = hpa
                if (i[0] == name):
                    final.append(hpi)
            if (len(child_hpa) < 2):
                while (len(child_node) > 0):
                    child = child_node[0]
                    i3 = child
                    hpi = self._listoutsinglebraches(i3, name, hpi, final, lsvarname, childeffect, lsparentname, con)
                    hpi = hpa + hpi
                    child_node.remove(child)
                    if (i[0] == name):
                        final.append(hpi)
            else:
                count = 1
                numberChild = len(child_node)
                for child in child_node:
                    i3 = child
                    hpi = self._listoutsinglebraches(i3, name, hpi, final, lsvarname, childeffect, lsparentname, con)
                    count = count + 1
                    if (count < 2 or count > numberChild):
                        hpi = hpa + hpi
                if (i[0] == name):
                    final.append(hpi)
        else:
            #hpchild = [(x[0], x[1], True) for x in hp if x[0] == i[0]]
            temp = hp[i[0]]
            temp.iskeep = True
            temp.rebuild()
            #hpchild = [temp]
            hpi.append(temp)
        return hpi

    def listoutAllBranches(self, lsvarname, childeffect, lsparentname, con: ConditionalSpace) -> List[
        SearchSpace]:
        #hp = copy.deepcopy(self._hyperparameters)
        hp=self._hyperparameters
        hpi = []
        returnList = []
        childList = [x[1] for x in childeffect]
        lsvarname=list(lsvarname)
        hpb_name = [x for x in hp if x not in (lsvarname + childList)]
        norelationLst=[]
        for i in hpb_name:
            norelationLst.append([[hp[i]]])
        #norelationLst = []
        #if (len(hpb) > 0):
         #   for i in hpb:
                # print(i)
         #       norelationLst.append(i)
        child_hpas = [x for x in lsvarname if x not in childList]
        # print(hpa)
        for child_hpa in child_hpas:
            # print(child_hpa)
            for newi in [x for x in lsparentname if x[0] == child_hpa]:
                # print(newi)
                hpi = []
                hpi = self._listoutsinglebraches(newi, newi[0], hpi, returnList, lsvarname, childeffect, lsparentname,
                                            con)

        count = 1
        final = []
        MixList=[]
        for child_hpa in child_hpas:
            tempList=[]
            for i1 in returnList:
                in1Lst = False
                for i2 in i1:
                    # i2=SearchSpace(i2)
                    if (i2.var_name[0] == child_hpa):
                        in1Lst = True
                if (in1Lst == True):
                    tempList.append(i1)
            MixList.append(tempList)
        MixList=MixList+norelationLst
        #MixList
        if(len(MixList)>1):
            final=list(itertools.product(*MixList))
        elif(len(MixList)==1):
            final=MixList[0]
        else:
            pass

        return final

    def combinewithconditional(self, cons: ConditionalSpace, ifAllSolution=True) -> List[SearchSpace]:
        listParam = OrderedDict()
        ordNo = 0
        for i, param in self._hyperparameters.items():
            listParam[i] = param.bounds[0]
            if len(param.id_N) >= 1:
                self._OrgLevels[ordNo] = param.bounds[0]
                ordNo += 1
                self.dim = ordNo
        for i, con in cons.conditional.items():
            if con[0] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[0]))
            else:
                if con[2] not in listParam[con[1]]:
                    raise TypeError("Value  '%s' doesnt exists" %
                                    str(con[2]))
            if con[1] not in listParam.keys():
                raise TypeError("Hyperparameter '%s' is not exists in current ConfigSpace" %
                                str(con[1]))
        lsSearchSpace, lsParentName = [], []
        lsParentName = []
        lsChildEffect = []
        lsFinalSP = []
        lsVarName = []
        for i, con in cons.conditional.items():
            lsParentName.append([con[1], con[2]])
            lsVarName.append([con[1]])
            lsChildEffect.append([str(con[1]) + "_" + str(con[2]), con[0]])
        lsParentName = [t for t in (set(tuple(i) for i in lsParentName))]
        lsVarName = np.unique(np.array(lsVarName))
        if (ifAllSolution == True):
            for a1 in lsVarName:
                noCon = self._hyperparameters[a1].bounds[0]
                for a2 in lsParentName:
                    noCon = [item for item in noCon if item not in a2[1]]
                # print(noCon)
                if(len(noCon)>0):
                    if(len(noCon)<2):
                        lsParentName.append(tuple([a1, noCon[0]]))
                    else:
                        lsParentName.append(tuple([a1, tuple(noCon)]))
                    #','.join([str(elem) for elem in noCon])
                #for a3 in noCon:
                #    lsParentName.append(tuple([a1, a3]))
        lsSearchSpace = self.listoutAllBranches(lsVarName, lsChildEffect, lsParentName, cons)
        for searchSpace in lsSearchSpace:
            FinalSP= OrderedDict()
            for group in searchSpace:
                for item in group:
                    if (item.iskeep==True):
                        FinalSP[item.var_name[0]]= item
                        if 'space' not in locals():
                            space=item
                            #dang lam do
                        else:
                            space= space*item
                 #       searchSpace.pop(name)
            lsFinalSP.append(space)
            del space
        return lsFinalSP

if __name__ == '__main__':
    np.random.seed(1)
    cs = ConfigSpace()

    con = ConditionalSpace("test")

    dataset = NominalSpace(["anh"], "dataset")
    alg_name = NominalSpace(['SVM', 'LinearSVC', 'RF', 'DTC', 'KNN', 'Quadratic'], 'alg_name')
    # dataset = NominalSpace( [datasetStr],"dataset")
    cs.add_multiparameter([dataset, alg_name])
    ##module1
    ####Missingvalue
    missingvalue = NominalSpace(['imputer', 'fillna'], 'missingvalue')
    strategy = NominalSpace(["mean", "median", "most_frequent", "constant"], 'strategy')
    cs.add_multiparameter([missingvalue, strategy])
    con.addConditional(strategy, missingvalue, ['imputer'])
    ####ENCODER
    encoder = NominalSpace(['OneHotEncoder', 'dummies'], 'encoder')
    OneHotEncoder_isUse = NominalSpace([True, False], 'isUse')
    dummy_na = NominalSpace([True, False], 'dummy_na')
    drop_first = NominalSpace([True, False], 'drop_first')
    cs.add_multiparameter([encoder, OneHotEncoder_isUse, dummy_na, drop_first])
    con.addConditional(OneHotEncoder_isUse, encoder, ['OneHotEncoder'])
    con.addMutilConditional([dummy_na, drop_first], encoder, ['dummies'])
    ###ReScaling
    rescaling = NominalSpace(['MinMaxScaler', 'StandardScaler', 'RobustScaler'], 'rescaling')
    ####IMBALANCED
    random_state = NominalSpace([27], 'random_state')
    imbalance = NominalSpace(['NONE', 'SMOTE', 'SMOTENC', 'SMOTETomek', 'SMOTEENN', ], 'imbalance')
    sampling_strategy_SMOTE = NominalSpace(['minority', 'not minority', 'not majority', 'all'],
                                           'sampling_strategy_SMOTE')
    k_neighbors_SMOTE = OrdinalSpace([3, 20], 'k_neighbors_SMOTE')
    categorical_features = NominalSpace([True], 'categorical_features')
    sampling_strategy_SMOTENC = NominalSpace(['minority', 'not minority', 'not majority', 'all'],
                                             'sampling_strategy_SMOTENC')
    k_neighbors_SMOTENC = OrdinalSpace([3, 20], 'k_neighbors_SMOTENC')

    sampling_strategy2 = NominalSpace(['minority', 'not minority', 'not majority', 'all'],
                                      'sampling_strategy_SMOTETomek')

    sampling_strategy1 = NominalSpace(['minority', 'not minority', 'not majority', 'all'], 'sampling_strategy_SMOTEENN')
    cs.add_multiparameter(
        [rescaling, random_state, imbalance, sampling_strategy_SMOTE, k_neighbors_SMOTE, categorical_features,
         sampling_strategy_SMOTENC, k_neighbors_SMOTENC, sampling_strategy2, sampling_strategy1])
    con.addMutilConditional([random_state, sampling_strategy_SMOTE, k_neighbors_SMOTE], imbalance, ['SMOTE'])
    con.addMutilConditional([random_state, categorical_features, sampling_strategy_SMOTENC, k_neighbors_SMOTENC],
                            imbalance, ['SMOTENC'])
    con.addConditional(sampling_strategy2, imbalance, ['SMOTETomek'])
    con.addConditional(sampling_strategy1, imbalance, ['SMOTEENN'])
    # Module 2
    FeaturePrepocessing = NominalSpace(['None', 'FastICA', 'PCA', 'SelectPercentile'], 'FeaturePrepocessing')
    ##FastICA
    n_components = OrdinalSpace([2, 50], 'n_components')
    algorithm_FastICA = NominalSpace(['parallel', 'deflation'], 'algorithm_FastICA')
    whiten = NominalSpace([True, False], 'whiten')
    fun = NominalSpace(['logcosh', 'exp', 'cube'], 'fun')
    tol_FastICA = ContinuousSpace([0.0, 1], 'tol_FastICA')
    cs.add_multiparameter([FeaturePrepocessing, n_components, algorithm_FastICA, whiten, fun, tol_FastICA])
    con.addMutilConditional([n_components, algorithm_FastICA, whiten, fun, tol_FastICA], FeaturePrepocessing,
                            ['FastICA'])
    ##PCA
    # n_components = OrdinalSpace([2, 50],'n_components')
    svd_solver = NominalSpace(['auto', 'full', 'arpack', 'randomized'], 'svd_solver')
    copy = NominalSpace([True, False], 'copy')
    whiten_PCA = NominalSpace([True, False], 'whiten_PCA')
    iterated_power = OrdinalSpace([1, 50], 'iterated_power')
    tol_PCA = ContinuousSpace([0.0, 1], 'tol_PCA')
    cs.add_multiparameter([svd_solver, copy, whiten_PCA, iterated_power, tol_PCA])
    con.addMutilConditional([svd_solver, copy, whiten_PCA, iterated_power, tol_PCA], FeaturePrepocessing, ['PCA'])
    ##SelectPercentile
    score_func = NominalSpace(['f_classif', 'f_regression', 'mutual_info_classif'], 'score_func')
    percentile = OrdinalSpace([0, 100], 'percentile')
    cs.add_multiparameter([score_func, percentile])
    con.addMutilConditional([score_func, percentile], FeaturePrepocessing, ['SelectPercentile'])
    # MODULE3
    # SVM
    probability = NominalSpace(['True', 'False'], 'probability')
    C = ContinuousSpace([1e-2, 100], 'C')
    kernel = NominalSpace(["linear", "rbf", "poly", "sigmoid"], 'kernel')
    coef0 = ContinuousSpace([0.0, 10.0], 'coef0')
    degree = OrdinalSpace([1, 5], 'degree')
    shrinking = NominalSpace(['True', 'False'], "shrinking")
    gamma = NominalSpace(['auto', 'value'], "gamma")
    gamma_value = ContinuousSpace([1e-2, 100], 'gamma_value')
    cs.add_multiparameter([probability, C, kernel, coef0, degree, shrinking, gamma, gamma_value])
    con.addMutilConditional([probability, C, kernel, coef0, degree, shrinking, gamma, gamma_value], alg_name, ['SVM'])
    # 'name': 'LinearSVC',
    penalty = NominalSpace(["l1", "l2"], 'penalty')
    # "loss" : hp.choice('loss',["hinge","squared_hinge"]),
    dual = NominalSpace([False], 'dual')
    tol = ContinuousSpace([0.0, 1], 'tol')
    multi_class = NominalSpace(['ovr', 'crammer_singer'], 'multi_class')
    fit_intercept = NominalSpace([False], 'fit_intercept')
    C_Lin = ContinuousSpace([1e-2, 100], 'C_Lin')
    cs.add_multiparameter([penalty, dual, tol, multi_class, fit_intercept, C_Lin])
    con.addMutilConditional([penalty, dual, tol, multi_class, fit_intercept, C_Lin], alg_name, ['LinearSVC'])
    # elif (alg_nameStr == "RF"):
    n_estimators = OrdinalSpace([5, 2000], "n_estimators")
    criterion = NominalSpace(["gini", "entropy"], "criterion")
    max_depth = OrdinalSpace([10, 200], "max_depth")
    max_features = NominalSpace(['auto', 'sqrt', 'log2', 'None'], "max_features")
    min_samples_split = OrdinalSpace([2, 200], "min_samples_split")
    min_samples_leaf = OrdinalSpace([2, 200], "min_samples_leaf")
    bootstrap = NominalSpace([True, False], "bootstrap")
    class_weight = NominalSpace(['balanced', 'None'], "class_weight")
    cs.add_multiparameter(
        [n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split, bootstrap,
         class_weight])
    con.addMutilConditional([n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split,
                             bootstrap, class_weight], alg_name, ['RF'])
    # 'name': 'DTC',
    splitter = NominalSpace(['best', 'random'], "splitter")
    criterion_dtc = NominalSpace(["gini", "entropy"], 'criterion_dtc')
    max_depth_dtc = OrdinalSpace([10, 200], 'max_depth_dtc')
    max_features_dtc = NominalSpace(['auto', 'sqrt', 'log2', 'None'], 'max_features_dtc')
    min_samples_split_dtc = OrdinalSpace([2, 200], 'min_samples_split_dtc')
    min_samples_leaf_dtc = OrdinalSpace([2, 200], 'min_samples_leaf_dtc')
    class_weight_dtc = NominalSpace(['balanced', 'None'], "class_weight_dtc", )
    # ccp_alpha = ContinuousSpace([0.0, 1.0],'ccp_alpha')
    cs.add_multiparameter([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                           min_samples_leaf_dtc, class_weight_dtc])
    con.addMutilConditional([splitter, criterion_dtc, max_depth_dtc, max_features_dtc, min_samples_split_dtc,
                             min_samples_leaf_dtc, class_weight_dtc], alg_name, ['DTC'])
    # elif (alg_nameStr == 'KNN'):
    n_neighbors = OrdinalSpace([5, 200], "n_neighbors")
    weights = NominalSpace(["uniform", "distance"], "weights")
    algorithm = NominalSpace(['auto', 'ball_tree', 'kd_tree', 'brute'], "algorithm")
    leaf_size = OrdinalSpace([1, 200], "leaf_size")
    p = OrdinalSpace([1, 200], "p")
    metric = NominalSpace(['euclidean', 'manhattan', 'chebyshev', 'minkowski'], "metric")
    # p_sub_type = name
    cs.add_multiparameter([n_neighbors, weights, algorithm, leaf_size, p, metric])
    con.addMutilConditional([n_neighbors, weights, algorithm, leaf_size, p, metric], alg_name, ['KNN'])

    # 'name': 'Quadratic',
    reg_param = ContinuousSpace([1e-2, 1], 'reg_param')
    store_covariance = NominalSpace(['True', 'False'], 'store_covariance')
    tol_qua = ContinuousSpace([0.0, 1], 'tol_qua')
    cs.add_multiparameter([reg_param, store_covariance, tol_qua])
    con.addMutilConditional([reg_param, store_covariance, tol_qua], alg_name, ['Quadratic'])
    lsSpace = cs.combinewithconditional(con, ifAllSolution=True)

    # lsSpace = cs.combinewithconditional(con)
    # print(lsSpace.sampling(10))
    orgDim = len(cs)
    # space1 = [Anh, kernel, I, Test]
    for ls in lsSpace:
        print("ratio:", float(len(ls) / orgDim))
        print(ls.sampling())
    # space = N * kernel * Anh * I
    # print(space.sampling(10))
    # from mipego import mipego

    # print((C * 2).var_name)
    # print((N * 3).sampling(2))
