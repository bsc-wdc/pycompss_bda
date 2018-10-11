#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
__copyright__ = '2018 Barcelona Supercomputing Center (BSC-CNS)'

import mmap
import os
from itertools import islice
from time import time
from uuid import uuid4

import numpy as np
from exceptions import AttributeError
from pycompss.api.api import compss_barrier as barrier
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import *
from pycompss.api.task import task
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC


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
        self._npartitions = 0
        self._tol = 0
        self._last_W = 0
        self._clf = None
        self._clf_params = None
        self._kernel_f = None

    def fit(self, path=None, data_format="csv", n_features=None,
            cascade_arity=2, n_partitions=4,
            cascade_iterations=5, tol=10 ** -3, C=1.0, kernel="rbf",
            gamma="auto"):

        try:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel[kernel])
        except AttributeError:
            self._kernel_f = getattr(self, CascadeSVM.name_to_kernel["rbf"])

        assert (gamma is "auto" or type(gamma) == float or type(
            float(gamma)) == float), "Gamma is not a valid float"
        assert (kernel is None or kernel in self.name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (
                kernel, self.name_to_kernel.keys())
        assert (C is None or type(C) == float or type(float(C)) == float), \
            "Incorrect C type [%s], type : %s" % (C, type(C))
        assert cascade_arity > 1, "Cascade arity must be greater than 1"
        assert cascade_iterations > 0, "Max iterations must be greater than 0"

        self._cascade_arity = cascade_arity
        self._max_iterations = cascade_iterations
        self._npartitions = n_partitions
        self._tol = tol
        self._last_W = 0
        self._clf = None
        self._clf_params = {"gamma": gamma, "C": C, "kernel": kernel}

        self.read_time = time()
        self.total_time = time()

        # if data_format == "libsvm":
        assert n_features > 0 or data_format != "libsvm" # "Number of features is required when using libsvm format"
        files = os.listdir(path)

        if not n_features:
            n_features = self._count_features(os.path.join(path, files[0]))

        partitions = self._read_dir(files, path, data_format, n_features)

        # Uncomment to measure read time
        # barrier()
        self.read_time = time() - self.read_time
        self.fit_time = time()

        self._cascade_fit(partitions)

        barrier()

        self.fit_time = time() - self.fit_time

        self.total_time = time() - self.total_time

    def score(self, X, y):
        if self._clf:
            return self._clf.score(X, y)
        else:
            raise Exception(
                "Model %s has not been initialized. Try calling fit first.")

    def _read_dir(self, files, path, data_format, n_features):

        if self._clf_params["gamma"] == "auto":
            self._clf_params["gamma"] = 1. / n_features

        self._npartitions = len(files)

        partitions = []

        for f in files:
            partitions.append(read_partition(os.path.join(path, f), data_format=data_format,
                               n_features=n_features))

        return partitions

    def _cascade_fit(self, partitions):
        feedback = None

        while self.iterations < self._max_iterations and not self.converged:
            feedback = self._cascade_iteration(partitions, feedback)

    def _cascade_iteration(self, partitions, feedback):
        q = []

        # first layer
        for partition in partitions:
            data = filter(None, [partition, feedback])
            q.append(train(False, *data, **self._clf_params))

        # reduction
        while len(q) > self._cascade_arity:
            data = q[:self._cascade_arity]
            del q[:self._cascade_arity]

            q.append(train(False, *data, **self._clf_params))

        # last layer
        final = compss_wait_on(train(True, *q, **self._clf_params))

        sv, sl, si, self._clf = final
        feedback = (sv, sl, si)
        self.iterations += 1

        self._check_convergence_and_update_w(feedback[0], feedback[1])
        print("Iteration %s of %s. \n" % (
            self.iterations, self._max_iterations))

        return feedback

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
            print("\tComputed W %s" % W)

            if self._last_W:
                delta = np.abs((W - self._last_W) / self._last_W)
                if delta < self._tol:
                    print("     Converged with delta: %s " % delta)
                    self.converged = True
                else:
                    print("\tNo convergence with delta: %s " % delta)
            else:
                print("\tFirst iteration, not testing convergence.")
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
def read_partition(filename, data_format=None, n_features=None):
    if data_format == "libsvm":
        X, y = load_svmlight_file(filename, n_features)

        X = X.toarray()
    else:
        with open(filename) as f:
            vecs = np.genfromtxt(islice(f, None, None), delimiter=",")

        X, y = vecs[:, :-1], vecs[:, -1]

    idx = np.array([uuid4().int for _ in range(X.shape[0])])

    return X, y, idx


def merge(*args):
    sv = np.concatenate(zip(*args)[0])
    sl = np.concatenate(zip(*args)[1])
    si = np.concatenate(zip(*args)[2])

    si, uniques = np.unique(si, return_index=True)
    sv = sv[uniques]
    sl = sl[uniques]

    return sv, sl, si


import argparse
import numpy as np
# from csvm_pycompss import CascadeSVM
import csv
import os
from sklearn.datasets import load_svmlight_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cr", "--centralized_read",
                        help="read the whole CSV file at the master",
                        action="store_true")
    parser.add_argument("--libsvm", help="read files in libsvm format",
                        action="store_true")
    parser.add_argument("-dt", "--detailed_times",
                        help="get detailed execution times (read and fit)",
                        action="store_true")
    parser.add_argument("-k", metavar="KERNEL", type=str,
                        help="linear or rbf (default is rbf)",
                        choices=["linear", "rbf"], default="rbf")
    parser.add_argument("-a", metavar="CASCADE_ARITY", type=int,
                        help="default is 2", default=2)
    parser.add_argument("-n", metavar="N_CHUNKS", type=int,
                        help="number of chunks in which to divide the dataset (default is 4)",
                        default=4)
    parser.add_argument("-i", metavar="MAX_ITERATIONS", type=int,
                        help="default is 5", default=5)
    parser.add_argument("-g", metavar="GAMMA", type=float,
                        help="(only for rbf kernel) default is 1 / n_features",
                        default=None)
    parser.add_argument("-c", metavar="C", type=float, help="default is 1",
                        default=1)
    parser.add_argument("-f", metavar="N_FEATURES", type=int,
                        help="mandatory if --libsvm option is used and train_data is a directory (optional otherwise)",
                        default=None)
    parser.add_argument("-t", metavar="TEST_FILE_PATH",
                        help="test CSV file path", type=str, required=False)
    parser.add_argument("-o", metavar="OUTPUT_FILE_PATH",
                        help="output file path", type=str, required=False)
    parser.add_argument("-nd", metavar="N_DATASETS", type=int,
                        help="number of times to load the dataset", default=1)
    parser.add_argument("--convergence", help="check for convergence",
                        action="store_true")
    parser.add_argument("--dense", help="use dense data structures",
                        action="store_true")
    parser.add_argument("train_data",
                        help="CSV file or directory containing CSV files (if a directory is provided N_CHUNKS is ignored)",
                        type=str)
    args = parser.parse_args()

    train_data = args.train_data

    csvm = CascadeSVM()

    if not args.g:
        gamma = "auto"
    else:
        gamma = args.g

    if args.centralized_read:
        if args.libsvm:
            x, y = load_svmlight_file(train_data)
        else:
            train = np.loadtxt(train_data, delimiter=",", dtype=float)

            x = train[:, :-1]
            y = train[:, -1]

        for _ in range(args.nd):
            csvm.load_data(X=x, y=y, kernel=args.k, C=args.c,
                           cascade_arity=args.a, n_chunks=args.n, gamma=gamma,
                           cascade_iterations=args.i, force_dense=args.dense)

    elif args.libsvm:
        csvm.fit(path=train_data, data_format="libsvm", n_features=args.f,
                 kernel=args.k, C=args.c,
                 cascade_arity=args.a, n_partitions=args.n, gamma=gamma,
                 cascade_iterations=args.i)

    else:
        csvm.fit(path=train_data, n_features=args.f, kernel=args.k, C=args.c,
                 cascade_arity=args.a,
                 n_partitions=args.n, gamma=gamma, cascade_iterations=args.i)

    out = [args.k, args.a, args.n, csvm._clf_params["gamma"], args.c,
           csvm.iterations, csvm.converged,
           csvm.read_time, csvm.fit_time, csvm.total_time]

    if os.path.isdir(train_data):
        n_files = os.listdir(train_data)
        out.append(len(n_files))

    if args.t:
        if args.libsvm:
            testx, testy = load_svmlight_file(args.t, args.f)

            if args.dense:
                testx = testx.toarray()

            out.append(csvm.score(testx, testy))
        else:
            test = np.loadtxt(args.t, delimiter=",", dtype=float)
            out.append(csvm.score(test[:, :-1], test[:, -1]))

    if args.o:
        with open(args.o, "ab") as f:
            wr = csv.writer(f)
            wr.writerow(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
