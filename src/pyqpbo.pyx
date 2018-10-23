###############################################################
# Python bindings for QPBO algorithm by Vladimir Kolmogorov.
#
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# More infos on my blog: peekaboo-vision.blogspot.com

cimport cython

import numpy as np
cimport numpy as np
from libcpp cimport bool

np.import_array()

cdef extern from "stdlib.h":
    void srand(unsigned int seed)

ctypedef int NodeId
ctypedef int EdgeId

cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(int node_num_max, int edge_num_max) nogil
        bool Save(char*filename, int format=0) nogil
        bool Load(char*filename) nogil
        void Reset() nogil
        NodeId AddNode(int num) nogil
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1) nogil
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10,
                               REAL E11) nogil
        void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, REAL E00, REAL E01,
                             REAL E10, REAL E11) nogil
        void MergeParallelEdges() nogil
        int GetLabel(NodeId i)
        void Solve() nogil
        void ComputeWeakPersistencies() nogil
        bool Improve() nogil

cdef class QPBOFloat:
    cdef QPBO[float] *_qpbo

    def __cinit__(self, int node_num_max, int edge_num_max):
        self._qpbo = new QPBO[float](node_num_max, edge_num_max)

        if self._qpbo is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._qpbo is not NULL:
            del self._qpbo

    def reset(self):
        with nogil:
            self._qpbo.Reset()

    def add_nodes(self, int num):
        return self._qpbo.AddNode(num)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids,
                        np.ndarray[np.float32_t, ndim=2, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddUnaryTerm(node_ids[i], weights[i, 0], weights[i, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self,
                           np.ndarray[np.int32_t, ndim=2, mode='c'] node_ids,
                           np.ndarray[np.float32_t, ndim=3, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddPairwiseTerm(
                node_ids[i, 0], node_ids[i, 1],
                weights[i, 0, 0], weights[i, 0, 1],
                weights[i, 1, 0], weights[i, 1, 1]
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_structured_unary_terms(
            self,
            np.ndarray[np.float32_t, ndim=2, mode='c'] weights
    ):
        for i in xrange(weights.shape[1]):
            self._qpbo.AddUnaryTerm(i, weights[0, i], weights[1, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_structured_pairwise_terms(self,
                           np.ndarray[np.int32_t, ndim=1, mode='c'] strides,
                           np.ndarray[np.float32_t, ndim=4, mode='c'] weights):
        cdef float weight_0_0
        cdef float weight_0_1
        cdef float weight_1_0
        cdef float weight_1_1

        for i in xrange(weights.shape[3]):
            for n in xrange(strides.shape[0]):
                stride = strides[n]

                weight_0_0 = weights[n, 0, 0, i]
                weight_0_1 = weights[n, 0, 1, i]
                weight_1_0 = weights[n, 1, 0, i]
                weight_1_1 = weights[n, 1, 1, i]

                if weight_0_0 == 0.0 and \
                        weight_0_1 == 0.0 and \
                        weight_1_0 == 0.0 and \
                        weight_1_1 == 0.0:
                    # If all weights are zero, we don't include the term.
                    # Note that we also assume that if i is near the border
                    # and i + stride would result in a illegal node then
                    # all weights will be zero. So this also takes care of
                    # out-of-bounds neighbours.
                    continue

                self._qpbo.AddPairwiseTerm(
                    i, i + stride,
                    weight_0_0, weight_0_1,
                    weight_1_0, weight_1_1
                )

    def merge_parallel_edges(self):
        with nogil:
            self._qpbo.MergeParallelEdges()

    def solve(self):
        with nogil:
            self._qpbo.Solve()

    def compute_weak_persistencies(self):
        with nogil:
            self._qpbo.ComputeWeakPersistencies()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_labels(self, np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = node_ids.shape[0]

        cdef np.ndarray[np.int32_t, ndim=1] result = \
            np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)

        cdef int *result_ptr = <int*> result.data
        for i in xrange(node_ids.shape[0]):
            result_ptr[i] = self._qpbo.GetLabel(node_ids[i])

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_structured_labels(self, int num):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = num

        cdef np.ndarray[np.int32_t, ndim=1] result = \
            np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)

        cdef int *result_ptr = <int*> result.data
        for i in xrange(num):
            result_ptr[i] = self._qpbo.GetLabel(i)

        return result

cdef class QPBODouble:
    cdef QPBO[double] *_qpbo

    def __cinit__(self, int node_num_max, int edge_num_max):
        self._qpbo = new QPBO[double](node_num_max, edge_num_max)

        if self._qpbo is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._qpbo is not NULL:
            del self._qpbo

    def reset(self):
        with nogil:
            self._qpbo.Reset()

    def add_nodes(self, int num):
        return self._qpbo.AddNode(num)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids,
                        np.ndarray[np.float64_t, ndim=2, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddUnaryTerm(node_ids[i], weights[i, 0], weights[i, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self,
                           np.ndarray[np.int32_t, ndim=2, mode='c'] node_ids,
                           np.ndarray[np.float64_t, ndim=3, mode='c'] weights):
        for i in xrange(node_ids.shape[0]):
            self._qpbo.AddPairwiseTerm(
                node_ids[i, 0], node_ids[i, 1],
                weights[i, 0, 0], weights[i, 0, 1],
                weights[i, 1, 0], weights[i, 1, 1]
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_structured_unary_terms(
            self,
            np.ndarray[np.float64_t, ndim=2, mode='c'] weights
    ):
        for i in xrange(weights.shape[1]):
            self._qpbo.AddUnaryTerm(i, weights[0, i], weights[1, i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_structured_pairwise_terms(self,
                           np.ndarray[np.int32_t, ndim=1, mode='c'] strides,
                           np.ndarray[np.float64_t, ndim=4, mode='c'] weights):
        cdef double weight_0_0
        cdef double weight_0_1
        cdef double weight_1_0
        cdef double weight_1_1

        for i in xrange(weights.shape[3]):
            for n in xrange(strides.shape[0]):
                stride = strides[n]

                weight_0_0 = weights[n, 0, 0, i]
                weight_0_1 = weights[n, 0, 1, i]
                weight_1_0 = weights[n, 1, 0, i]
                weight_1_1 = weights[n, 1, 1, i]

                if weight_0_0 == 0.0 and \
                        weight_0_1 == 0.0 and \
                        weight_1_0 == 0.0 and \
                        weight_1_1 == 0.0:
                    # If all weights are zero, we don't include the term.
                    # Note that we also assume that if i is near the border
                    # and i + stride would result in a illegal node then
                    # all weights will be zero. So this also takes care of
                    # out-of-bounds neighbours.
                    continue

                self._qpbo.AddPairwiseTerm(
                    i, i + stride,
                    weight_0_0, weight_0_1,
                    weight_1_0, weight_1_1
                )

    def merge_parallel_edges(self):
        with nogil:
            self._qpbo.MergeParallelEdges()

    def solve(self):
        with nogil:
            self._qpbo.Solve()

    def compute_weak_persistencies(self):
        with nogil:
            self._qpbo.ComputeWeakPersistencies()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_labels(self, np.ndarray[np.int32_t, ndim=1, mode='c'] node_ids):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = node_ids.shape[0]

        cdef np.ndarray[np.int32_t, ndim=1] result = \
            np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)

        cdef int *result_ptr = <int*> result.data
        for i in xrange(node_ids.shape[0]):
            result_ptr[i] = self._qpbo.GetLabel(node_ids[i])

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_structured_labels(self, int num):
        cdef np.npy_intp result_shape[1]
        result_shape[0] = num

        cdef np.ndarray[np.int32_t, ndim=1] result = \
            np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)

        cdef int *result_ptr = <int*> result.data
        for i in xrange(num):
            result_ptr[i] = self._qpbo.GetLabel(i)

        return result
