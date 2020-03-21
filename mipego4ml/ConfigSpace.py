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
        hp = copy.deepcopy(self._hyperparameters)
        hpi = []
        returnList = []
        childList = [x[1] for x in childeffect]
        lsvarname=list(lsvarname)
        hpb_name = [x for x in hp if x not in (lsvarname + childList)]
        norelationLst=[]
        for i in hpb_name:
            norelationLst.append(hp[i])
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
        child_hpa = child_hpas[0]
        count = 1
        final = []
        temp1Lst = []
        temp2Lst = []
        for i1 in returnList:
            in1Lst = False
            for i2 in i1:
                # i2=SearchSpace(i2)
                if (i2.var_name[0] == child_hpa):
                    in1Lst = True
            if (in1Lst == True):
                temp1Lst.append(i1)
            else:
                temp2Lst.append(i1)

        for i1 in temp1Lst:
            # print('Firts',i)
            for i in temp2Lst:
                # print(count,i1, i)
                final.append(i1 + i + norelationLst)
                count = count + 1
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
            for item in searchSpace:
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
    dataset = NominalSpace(['anh'], "dataset")
    missingvalue = NominalSpace(["imputer", "fillna"], "missingvalue")
    strategy = NominalSpace(["mean", "median", "most_frequent", "constant"], "strategy")
    lv3 = NominalSpace([1, 2], "lv3")
    cs.add_multiparameter([dataset, missingvalue, strategy, lv3])
    con.addConditional(strategy, missingvalue, ["imputer"])
    con.addConditional(lv3, strategy, ["mean"])
    ####ENCODER
    encoder = NominalSpace(['OneHotEncoder', 'dummies'], 'encoder')
    OneHotEncoder_isUse = NominalSpace([True, False], 'isUse')
    dummy_na = NominalSpace([True, False], 'dummy_na')
    drop_first = NominalSpace([True, False], 'drop_first')
    cs.add_multiparameter([encoder, OneHotEncoder_isUse, dummy_na, drop_first])
    con.addConditional(OneHotEncoder_isUse, encoder, ['OneHotEncoder'])
    con.addMutilConditional([dummy_na, drop_first], encoder, ['dummies'])
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
