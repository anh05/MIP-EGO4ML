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

from mipego4ml.ConditionalSpace import ConditionalSpace
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
            # listParent(param.var_name)=(param.parent_name,param.parent_value)

        # self._update_infor()
        # self._set_index()
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

    def combinewithconditional(self, cons: ConditionalSpace, ifAllSolution=False) -> List[SearchSpace]:
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
                for a3 in noCon:
                    lsParentName.append(tuple([a1, a3]))
        for i, con in lsParentName:
            localHypers = copy.deepcopy(self._hyperparameters)
            temp = localHypers[i]
            temp.bounds = [tuple([con])]
            temp.iskeep = True
            temp.rebuild()
            localHypers[str(i)] = temp

            for a, b in lsChildEffect:
                localHypers[b].iskeep = False

            for a, b in lsChildEffect:
                if (a == (i + "_" + con)):
                    localHypers[b].iskeep = True

            lsSearchSpace.append(localHypers)
        for searchSpace in lsSearchSpace:
            FinalSP = OrderedDict()
            for name, item in searchSpace.items():
                if (item.iskeep == True):
                    FinalSP[name] = item
                    if 'space' not in locals():
                        space = item
                    else:
                        space = space * item
            #       searchSpace.pop(name)
            lsFinalSP.append(space)
            del space
        return lsFinalSP


if __name__ == '__main__':
    np.random.seed(1)
    C = ContinuousSpace([-5, 5])  # product of the same space
    Anh = NominalSpace(["SVM", "KNN", "ABC", "EM"], 'alg')
    kernel = OrdinalSpace([1, 100], 'kernel')
    I = OrdinalSpace([-20, 20], 'I1')
    RS = NominalSpace([27], 'randomstate')
    N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], "N")
    ABC = OrdinalSpace([-20, 20], 'ABC')
    # I3 = I * 3
    print(Anh.sampling())
    # print(kernel.sampling())
    # print(I3.var_name)

    # print(C.sampling(1, 'uniform'))

    # cartesian product of heterogeneous spaces
    con = ConditionalSpace("weare")
    con.addMutilConditional([I, kernel], Anh, "SVM")
    con.addConditional(ABC, Anh, ["SVM", "KNN"])
    con.addConditional(N, Anh, "KNN")
    cs = ConfigSpace()
    cs.add_multiparameter([Anh, C, kernel, I, N, RS, ABC])
    lsSpace = cs.combinewithconditional(con)
    print(RS.sampling(10))
    orgDim = len(cs)
    # space1 = [Anh, kernel, I, Test]
    for ls in lsSpace:
        print("ratio:", float(len(ls) / orgDim))
        print(ls.sampling())
    space = N * kernel * Anh * I
    print(space.sampling(10))
    # from mipego import mipego

    # print((C * 2).var_name)
    # print((N * 3).sampling(2))
