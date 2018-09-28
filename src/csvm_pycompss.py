from __future__ import print_function

import mmap
import os
from collections import deque
from exceptions import AttributeError
from itertools import islice
from time import time

import numpy as np
from pycompss.api.api import compss_barrier as barrier
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import *
from pycompss.api.task import task
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from uuid import uuid4


class CascadeSVM(object):
    name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self):

        self.iterations = 0
        self.converged = False

        self.read_time = 0
        self.fit_time = 0
        self.total_time = 0

        self._cascade_arity = 2
        self._max_iterations = 0
        self._nchunks = 0
        self._tol = 0
        self._last_W = 0
        self._clf = None
        self._clf_params = None
        self._kernel_f = None

    def fit(self, path=None, data_format="csv", n_features=None, cascade_arity=2, n_chunks=4,
            cascade_iterations=5, tol=10 ** -3, C=1.0, kernel="rbf", gamma="auto"):

        try:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel["rbf"])

        assert (gamma is "auto" or type(gamma) == float or type(float(gamma)) == float), "Gamma is not a valid float"
        assert (kernel is None or kernel in self.name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (kernel, self.name_to_kernel.keys())
        assert (C is None or type(C) == float or type(float(C)) == float), \
            "Incorrect C type [%s], type : %s" % (C, type(C))
        assert cascade_arity > 1, "Cascade arity must be greater than 1"
        assert cascade_iterations > 0, "Max iterations must be greater than 0"

        self._cascade_arity = cascade_arity
        self._max_iterations = cascade_iterations
        self._nchunks = n_chunks
        self._tol = tol
        self._last_W = 0
        self._clf = None
        self._clf_params = {"gamma": gamma, "C": C, "kernel": kernel}

        self.read_time = time()
        self.total_time = time()

        chunks = self._read_dir(path, data_format, n_features)

        # Uncomment to measure read time
        #barrier()

        self.read_time = time() - self.read_time
        self.fit_time = time()

        self._do_fit(chunks)

        barrier()

        self.fit_time = time() - self.fit_time

        self.total_time = time() - self.total_time

    def score(self, X, y):
        if self._clf:
            return self._clf.score(X, y)
        else:
            raise Exception("Model %s has not been initialized. Try calling fit first." % i)

    def _read_dir(self, path, data_format, n_features):
        files = os.listdir(path)

        if data_format == "libsvm":
            assert n_features > 0, "Number of features is required when using libsvm format"
        elif not n_features:
            n_features = self._count_features(os.path.join(path, files[0]))

        if self._clf_params["gamma"] == "auto":
            self._clf_params["gamma"] = 1. / n_features

        self._nchunks = len(files)

        chunks = []

        for f in files:
            chunks.append(read_chunk(os.path.join(path, f), data_format=data_format, n_features=n_features))

        return chunks

    def _do_fit(self, chunks):
        q = []
        feedback = None

        while self.iterations < self._max_iterations and not self.converged:

            # first level
            for chunk in chunks:
                data = filter(None, [chunk, feedback])
                q.append(train(False, *data, **self._clf_params))

            # reduction
            while q:
                n_elements = min(len(q), self._cascade_arity)

                data = q[:n_elements]
                del q[:n_elements]

                if q:
                    q.append(train(False, *data, **self._clf_params))
                else:
                    sv, sl, si, self._clf = compss_wait_on(train(True, *data, **self._clf_params))

                # delete partial results
                for d in data:
                    compss_delete_object(d)

            feedback = (sv, sl, si)
            self.iterations += 1

            self._check_convergence_and_update_w(feedback[0], feedback[1])
            print("Iteration %s of %s. \n" % (
                    self.iterations, self._max_iterations))

    def _lagrangian_fast(self, SVs, sl, coef):
        set_sl = set(sl)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = sl.copy()
        new_sl[sl == 0] = -1

        C1, C2 = np.meshgrid(coef, coef)
        L1, L2 = np.meshgrid(new_sl, new_sl)

        double_sum = C1 * C2 * L1 * L2 * self._kernel_f(SVs)
        double_sum = double_sum.sum()
        W = -0.5 * double_sum + coef.sum()

        return W

    def _rbf_kernel(self, x):
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._clf_params["gamma"])
        n = x.shape[0]
        K = x.dot(x.T) / sigmaq
        d = np.diag(K).reshape((n, 1))
        K = K - np.ones((n, 1)) * d.T / 2
        K = K - d * np.ones((1, n)) / 2
        K = np.exp(K)
        return K

    def _check_convergence_and_update_w(self, sv, sl):
        self.converged = False
        clf = self._clf
        print("Checking convergence:")

        if clf:
            W = self._lagrangian_fast(sv, sl, clf.dual_coef_)
            print("     Computed W %s" % W)

            if self._last_W:
                delta = np.abs((W - self._last_W) / self._last_W)
                if delta < self._tol:
                    print("     Converged with delta: %s " % delta)
                    self.converged = True
                else:
                    print("     No convergence with delta: %s " % delta)
            else:
                print("     First iteration, not testing convergence.")
            self._last_W = W
            print()

    @staticmethod
    def _count_features(filename):
        f = open(filename, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        line = buf.readline()
        features = len(line.split(",")) - 1
        f.close()
        return features

    @staticmethod
    def _linear_kernel(x1):
        return np.dot(x1, x1.T)


@task(returns=tuple)
def train(return_classifier, *args, **kwargs):
    if len(args) > 1:
        X, y, idx = merge(*args)
    else:
        X, y, idx = args[0]

    clf = SVC(random_state=1, **kwargs)
    clf.fit(X, y)

    sv = X[clf.support_]
    sl = y[clf.support_]
    idx = idx[clf.support_]

    if return_classifier:
        return sv, sl, idx, clf
    else:
        return sv, sl, idx


@task(filename=FILE, returns=tuple)
def read_chunk(filename, data_format=None, n_features=None):
    if data_format == "libsvm":
        X, y = load_svmlight_file(filename, n_features)

        X = X.toarray()
    else:
        with open(filename) as f:
            vecs = np.genfromtxt(islice(f, None, None), delimiter=",")

        X, y = vecs[:, :-1], vecs[:, -1]

    idx = np.array([uuid4().int for _ in xrange(X.shape[0])])

    return X, y, idx


def merge(*args):
    sv = np.concatenate([t[0] for t in args])
    sl = np.concatenate([t[1] for t in args])
    si = np.concatenate([t[2] for t in args])

    si, uniques = np.unique(si, return_index=True)
    sv = sv[uniques]
    sl = sl[uniques]

    return sv, sl, si
