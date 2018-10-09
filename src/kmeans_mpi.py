from time import time

import numpy as np
from mpi4py import MPI
from numpy import random as rng

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_arguments():
    """
    Parse command line arguments. Make the program generate
    a help message in case of wrong usage.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='An MPI Kmeans implementation.')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Pseudo-random seed. Default = 0'
                        )
    parser.add_argument('-n', '--num_points', type=int, default=100,
                        help='Number of points. Default = 100'
                        )
    parser.add_argument('-d', '--dimensions', type=int, default=2,
                        help='Number of dimensions. Default = 2'
                        )
    parser.add_argument('-c', '--num_centers', type=int, default=2,
                        help='Number of centers'
                        )
    parser.add_argument('-i', '--max_iterations', type=int, default=20,
                        help='Maximum number of iterations'
                        )
    parser.add_argument('-e', '--epsilon', type=float, default=1e-9,
                        help='Epsilon. Kmeans will stop when |old - new| < epsilon.'
                        )
    parser.add_argument('-l', '--lnorm', type=int, default=2, choices=[1, 2],
                        help='Norm for vectors'
                        )
    parser.add_argument('--distributed_read', action='store_true',
                        help='Boolean indicating if data should be read distributed.'
                        )

    return parser.parse_args()


def _calc_slices(X):
    """Calculate the slices of data for each process.

    Arguments:
        X : numpy.array, 2 dimensional

    Return:
        number of rows each process gets as a numpy array
    """

    n_rows = X.shape[0]
    slices = [n_rows // comm.size for _ in range(comm.size)]
    count = n_rows % comm.size
    for i in range(count):
        slices[i] += 1

    return np.array(slices, dtype=np.int64)


def _scatter_samples(X):
    if rank == 0:
        slices = _calc_slices(X)
        n_features = np.array(X.shape[1], dtype=int)
    else:
        slices = np.zeros(comm.size, dtype=np.int64)
        n_features = np.zeros(1, dtype=int)

    # Broadcast information for scatterv
    comm.Bcast(slices, root=0)
    comm.Bcast(n_features, root=0)

    # slices and pos for samples (2d has to be considered)
    data_slices = slices * n_features
    data_pos = np.array([sum(data_slices[:i]) for i in range(comm.size)],
                        dtype=np.float64)

    # number of rows for each process
    row_cnt = slices[comm.rank]

    # allocate memory for splitted samples and scatter it
    split_X = np.zeros((row_cnt, n_features.item()), dtype=np.float64)
    comm.Scatterv([X, data_slices, data_pos, MPI.DOUBLE], split_X, root=0)

    return split_X


def generate_fragment(num_points, dimensions, seed):
    if rank == 0:

        rng.seed(seed)

        matrix = init_board_random(num_points, dimensions, seed)
    else:
        matrix = None

    matrix = _scatter_samples(matrix)

    return matrix


def init_board_random(numV, dim, seed):
    np.random.seed(seed)
    return np.random.random((numV, dim))


def init_centers_random(dim, k, seed):
    np.random.seed(seed)
    m = np.random.random((k, dim))
    return m

def root_print(msg):
    if rank == 0:
        print(msg)

def has_converged(mu, oldmu, epsilon, iter, maxIterations):
    root_print("iter: " + str(iter))
    root_print("maxIterations: " + str(maxIterations))
    if oldmu != []:
        if iter < maxIterations:
            aux = [np.linalg.norm(oldmu[i] - mu[i]) for i in range(len(mu))]
            distancia = sum(aux)
            if distancia < epsilon * epsilon:
                root_print("Distancia_T: " + str(distancia))
                return True
            else:
                root_print("Distancia_F: " + str(distancia))
                return False
        else:
            root_print("Reached max number of iterations.")
            return True


def cluster_and_partial_sums(fragment, centers):
    partial_results = np.array(np.zeros(centers.shape))
    c = centers.shape[0]
    # Check if labels is an empty list
    labels = np.zeros(len(fragment), dtype=np.uint32)
    # Compute the big stuff
    associates = np.zeros(c)
    # Get the labels for each point
    for (i, point) in enumerate(fragment):
        distances = np.zeros(c)
        for (j, center) in enumerate(centers):
            distances[j] = np.linalg.norm(point - center)
        labels[i] = np.argmin(distances)
        associates[labels[i]] += 1
    # Add each point to its associate center
    for (i, point) in enumerate(fragment):
        partial_results[labels[i]] += point / associates[labels[i]]
    return partial_results, labels


def kmeans_mpi(num_points, dimensions, num_centers, max_iterations, seed,
               epsilon, distributed_read):
    if distributed_read:
        matrix = init_board_random(num_points // size, dimensions, seed)
    else:
        matrix = generate_fragment(num_points=num_points, dimensions=dimensions,
                                   seed=seed)

    if rank == 0:
        new_centers = init_centers_random(dimensions, num_centers, seed)

    else:
        new_centers = np.zeros((num_centers, dimensions))

    epsilon = epsilon
    comm.Bcast(new_centers, root=0)
    centers = []

    t0 = time()
    it = 0
    while not has_converged(new_centers, centers, epsilon, it, max_iterations):
        centers = new_centers

        if rank == 0:
            print("Iteration: %s [%.2f] (s)" % (it, time() - t0))
        t0 = time()

        partial_results, labels = cluster_and_partial_sums(matrix, centers)
        new_centers = np.array(np.zeros(centers.shape))

        partial_results = comm.gather(partial_results, root=0)
        if rank == 0:
            for partial in partial_results:
                # Mean of means, single step
                new_centers += partial / float(size)
        comm.Bcast(new_centers, root=0)
        if np.linalg.norm(centers - new_centers) < epsilon:
            return new_centers

        # Convergence criterion is not met, update centers
        # centers = new_centers
        it += 1


def main():
    args = parse_arguments()

    if rank == 0:
        print("Execution arguments:\n%s" % args)
        t0 = time()

    result = kmeans_mpi(num_points=args.num_points,
                        dimensions=args.dimensions,
                        num_centers=args.num_centers,
                        max_iterations=args.max_iterations,
                        seed=args.seed,
                        epsilon=args.epsilon,
                        distributed_read=args.distributed_read)

    if rank == 0:
        t1 = time()
        print("Total elapsed time: %s [points=%s]" % (t1 - t0, args.num_points))


if __name__ == "__main__":
    main()
