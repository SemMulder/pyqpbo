###############################################################
# Python bindings for QPBO algorithm by Vladimir Kolmogorov.
#
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# More infos on my blog: peekaboo-vision.blogspot.com

import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

np.import_array()

cdef extern from "stdlib.h":
    void srand(unsigned int seed)

ctypedef int NodeId
ctypedef int EdgeId

cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(int node_num_max, int edge_num_max)
        bool Save(char*filename, int format=0)
        bool Load(char*filename)
        void Reset()
        NodeId AddNode(int num)
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1)
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10,
                               REAL E11)
        void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, REAL E00, REAL E01,
                             REAL E10, REAL E11)
        int GetLabel(NodeId i)
        void Solve()
        void ComputeWeakPersistencies()
        bool Improve()

cdef class QPBOFloat:
    cdef QPBO[float] *_qpbo

    def __cinit__(self, int node_num_max, int edge_num_max):
        self._qpbo = new QPBO[float](node_num_max, edge_num_max)

        if self._qpbo is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._qpbo is not NULL:
            del self._qpbo

    def add_nodes(self, int num):
        return self._qpbo.AddNode(num)

    def add_unary_terms(self,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids,
                        np.ndarray[np.float32_t, ndim=2, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddUnaryTerm(node_ids[i], weights[i, 0], weights[i, 1])

    def add_pairwise_terms(self,
                           np.ndarray[np.int32_t, ndim=2, mode='c'] node_ids,
                           np.ndarray[np.float32_t, ndim=3, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddPairwiseTerm(
                node_ids[i, 0], node_ids[i, 1],
                weights[i, 0, 0], weights[i, 0, 1],
                weights[i, 1, 0], weights[i, 1, 1]
            )

    def solve(self):
        self._qpbo.Solve()

    def compute_weak_persistencies(self):
        self._qpbo.ComputeWeakPersistencies()

    def get_labels(self, np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = node_ids.shape[0]

        cdef np.ndarray[np.int32_t, ndim=1] result = \
            np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)

        cdef int *result_ptr = <int*> result.data
        for i in xrange(node_ids.shape[0]):
            result_ptr[i] = self._qpbo.GetLabel(node_ids[i])

        return result
