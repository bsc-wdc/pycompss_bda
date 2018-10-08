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
__author__ = 'Ramon Amela <ramon.amela@bsc.es>'
__copyright__ = '2018 Barcelona Supercomputing Center (BSC-CNS)'

from pycompss.api.task import task
from pycompss.api.parameter import *
import numpy as np

@task(returns=dict)
def mergeReduceTask(*data):
    reduce_value = data[0]
    for i in xrange(1, len(data)):
        reduce_value = reduceCenters(reduce_value, data[i])
    return reduce_value

def mergeReduce(function, data, chunk=50):
    while(len(data)) > 1:
        dataToReduce = data[:chunk]
        data = data[chunk:]
        data.append(mergeReduceTask(*dataToReduce))
    return data[0]

def init_board_gauss(numV, dim, K, seed):
    n = int(float(numV) / K)
    data = []
    np.random.seed(seed)
    for k in range(K):
        c = [np.random.uniform(-1, 1) for i in range(dim)]
        s = np.random.uniform(0.05, 0.5)
        for i in range(n):
            d = np.array([np.random.normal(c[j], s) for j in range(dim)])
            data.append(d)

    Data = np.array(data)[:numV]
    return Data


def init_board_random(numV, dim, seed):
    np.random.seed(seed)
    return np.random.random((numV, dim))


def init_random(dim, k, seed):
    np.random.seed(seed)
    m = np.random.random((k, dim))
    return m


@task(returns=dict)
def cluster_points_sum(XP, mu, ind):
    """
    For each point computes the nearest center.
    :param XP: Fragments of points
    :param mu: Centers
    :param ind: point first index
    :return: {mu_ind: [pointInd_i, ..., pointInd_n]}
    """
    dic = {}
    for x in enumerate(XP):
        bestmukey = min([(i[0], np.linalg.norm(x[1] - mu[i[0]]))
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        if bestmukey not in dic:
            dic[bestmukey] = [x[0] + ind]
        else:
            dic[bestmukey].append(x[0] + ind)
    clusters = {}
    for i in dic:
        p_idx = np.array(dic[i]) - ind 
        clusters[i] = (len(p_idx), np.sum(XP[p_idx], axis=0))
    return clusters

def reduceCenters(a, b):
    """
    Reduce method to sum the result of two partial_sum methods
    :param a: partial_sum {cluster_ind: (#points_a, sum(points_a))}
    :param b: partial_sum {cluster_ind: (#points_b, sum(points_b))}
    :return: {cluster_ind: (#points_a+#points_b, sum(points_a+points_b))}
    """
    for key in b:
        if key not in a:
            a[key] = b[key]
        else:
            a[key] = (a[key][0] + b[key][0], a[key][1] + b[key][1])
    return a


def has_converged(mu, oldmu, epsilon, iter, maxIterations):
    print "iter: " + str(iter)
    print "maxIterations: " + str(maxIterations)
    if oldmu != []:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < epsilon * epsilon:
                print "Distancia_T: " + str(distancia)
                return True
            else:
                print "Distancia_F: " + str(distancia)
                return False
        else:
            # detencion pq se ha alcanzado el maximo de iteraciones
            return True

@task(returns=list)
def genFragment(numv, dim, k, seed, mode="random"):
    if mode == "gauss":
        return init_board_gauss(numv, dim, k, seed)
    else:
        return init_board_random(numv, dim, seed)


def kmeans_frag(numV, k, dim, epsilon, maxIterations, numFrag):
    from pycompss.api.api import compss_wait_on
    from pycompss.api.api import barrier
    size = int(numV / numFrag)
    seed = 5
    startTime = time.time()
    X = [genFragment(size, dim, k, seed + i) for i in range(numFrag)]
    #barrier()
    print "Points generation Time {} (s)".format(time.time() - startTime)
    mu = init_random(dim, k, seed)
    oldmu = []
    n = 0
    startTime = time.time()
    
    while not has_converged(mu, oldmu, epsilon, n, maxIterations):
        oldmu = mu
        partialResult = []
        #clusters = []
        for f in xrange(numFrag):
            #cluster = cluster_points_partial(X[f], mu, f * size)
            #clusters.append(cluster)
            #partialResult.append(partial_sum(X[f], cluster, f * size)) 
            partialResult.append(cluster_points_sum(X[f], mu, f * size))
        #clusters = [cluster_points_partial(
        #    X[f], mu, f * size) for f in range(numFrag)]
        #partialResult = [partial_sum(
        #    X[f], clusters[f], f * size) for f in range(numFrag)]

        mu = mergeReduce(reduceCenters, partialResult, chunk=50)
        mu = compss_wait_on(mu)
        mu = [mu[c][1] / mu[c][0] for c in mu]
        while len(mu) < k:
            indP = np.random.randint(0, size)
            indF = np.random.randint(0, numFrag)
            mu.append(X[indF][indP])
        n += 1
        print "Iteration Time {} (s)".format(time.time() - startTime)
    print "Kmeans Time {} (s)".format(time.time() - startTime)
    
    return (n, mu)


if __name__ == "__main__":
    import sys
    import time

    numV = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])
    iterations = int(sys.argv[5])

    startTime = time.time()
    result = kmeans_frag(numV, k, dim, 1e-4, iterations, numFrag)
    print "Ellapsed Time {} (s)".format(time.time() - startTime)
