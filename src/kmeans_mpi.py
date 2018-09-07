import numpy as np

from math import ceil
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
    parser = argparse.ArgumentParser(description='A COMPSs-Redis Kmeans implementation.')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Pseudo-random seed. Default = 0'
                        )
    parser.add_argument('-n', '--numpoints', type=int, default=100,
                        help='Number of points. Default = 100'
                        )
    parser.add_argument('-d', '--dimensions', type=int, default=2,
                        help='Number of dimensions. Default = 2'
                        )
    parser.add_argument('-c', '--centers', type=int, default=2,
                        help='Number of centers'
                        )
    parser.add_argument('-f', '--fragments', type=int, default=10,
                        help='Number of fragments. Default = 10. Condition: fragments < points'
                        )
    parser.add_argument('-m', '--mode', type=str, default='uniform',
                        choices=['uniform', 'normal'],
                        help='Distribution of points. Default = uniform'
                        )
    parser.add_argument('-i', '--iterations', type=int, default=20,
                        help='Maximum number of iterations'
                        )
    parser.add_argument('-e', '--epsilon', type=float, default=1e-9,
                        help='Epsilon. Kmeans will stop when |old - new| < epsilon.'
                        )
    parser.add_argument('-l', '--lnorm', type=int, default=2, choices=[1, 2],
                        help='Norm for vectors'
                        )
    parser.add_argument('--plot_result', action='store_true',
                        help='Plot the resulting clustering (only works if dim = 2).'
                        )
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?'
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
    data_pos = np.array([sum(data_slices[:i]) for i in range(comm.size)], dtype=np.float64)

    # number of rows for each process
    row_cnt = slices[comm.rank]

    # allocate memory for splitted samples and scatter it
    split_X = np.zeros((row_cnt, n_features.item()), dtype=np.float64)
    comm.Scatterv([X, data_slices, data_pos, MPI.DOUBLE], split_X, root=0)

    return split_X


def _scatter_labels(y):
    if comm.rank == 0:
        slices = _calc_slices(y)

    else:
        slices = np.zeros(comm.size, dtype=np.int64)

    # Broadcast information for scatterv
    comm.Bcast(slices, root=0)

    # pos for labels
    pos = np.array([sum(slices[:i]) for i in range(comm.size)], dtype=int)

    # number of rows for each process
    row_cnt = slices[comm.rank]

    # allocate memory for splitted samples and scatter it
    split_y = np.zeros(row_cnt, dtype=np.int64)
    comm.Scatterv([y, tuple(slices), tuple(pos), MPI.INT_INT], split_y, root=0)

    return split_y


def _scatter_data(X, y):
    """ MPI method, scatters data X and labels y to all processes.

    Arguments:
        X : array_like, data, each row is a sample and each column a feature
        y : array_like, y[i] is the class label of X[i]

    Return:
        splitted X and y for each process as numpy arrays.

    """

    split_X = _scatter_samples(X)
    split_y = _scatter_labels(y)
    return split_X, split_y


def scatter_data(self, X, y):
    sX, sy = _scatter_data(X, y)

    return sX, sy

def plot(mat, labels, centers):
    import matplotlib.pyplot as plt
    plt.figure('Clustering')

    def color_wheel(i):
        l = ['red', 'purple', 'blue', 'cyan', 'green']
        return l[i % len(l)]

    idx = 0
    for (i, p) in enumerate(mat):
        col = color_wheel(labels[idx])
        plt.scatter(p[0], p[1], color=col)
        idx += 1
    for centre in centers:
        plt.scatter(centre[0], centre[1], color='black')
    import uuid
    plt.show()
    # plt.savefig('%s.png' % str(uuid.uuid4()))

def cluster_and_partial_sums(fragment, centers, norm):
    partial_results = np.array(np.zeros(centers.shape))
    c = centers.shape[0]
    # Check if labels is an empty list
    print("[%s] CP centers: %s" % (rank, centers))
    # print("[%s] Fragment: %s" % (rank, fragment))
    labels = np.zeros(len(fragment), dtype=np.uint32)
    # Compute the big stuff
    associates = np.zeros(c)
    # Get the labels for each point
    for (i, point) in enumerate(fragment):
        distances = np.zeros(c)
        for (j, center) in enumerate(centers):
            print("[%s] Point: %s" % (rank, point))
            print("[%s] Center: %s" % (rank, center))
            distances[j] = np.linalg.norm(point - center, norm)
        labels[i] = np.argmin(distances)
        print("New label: %s" % labels[i])
        associates[labels[i]] += 1
    # Add each point to its associate center
    print("[%s] Labels[i] %s" % (rank, labels))
    for (i, point) in enumerate(fragment):
        partial_results[labels[i]] += point / associates[labels[i]]
    return partial_results, labels


def main():
    args = parse_arguments()

    from time import time
    np.random.seed(int(time()))
    if rank == 0:

        rng.seed(5)
        # ind = rng.randint(0, len(X) - 1)
        matrix = np.array([np.array([1, 2]), np.array([-1, -2]), np.array([2, 1]),
                            np.array([-2, -1])])
        matrix = np.array([rng.random(args.dimensions) for _ in range(args.numpoints)])

        matrix = np.random.random((args.numpoints, args.dimensions))
        print("Created matrix: %s" % matrix)
        centers = np.array(
            [np.random.random(args.dimensions) for _ in range(args.centers)]
        )
    else:
        matrix = None
        centers = np.zeros((args.centers, args.dimensions))

    matrix = _scatter_samples(matrix)

    print("Matrix: %s" % matrix)
    # labels = np.zeros(int(ceil(matrix.shape[0] / size)), dtype=np.uint32)
    norm = args.lnorm
    epsilon = args.epsilon

    comm.Bcast(centers, root=0)

    for _ in range(args.iterations):
    # for _ in range(1):
        # centers = _scatter_labels(centers)
        # print("Rank [%s]:\nmatrix: %s\ncenters:%s" % (rank, matrix, centers))
        # labels = _scatter_labels(labels)
        print("[%s] centers:\n%s" % (rank, centers))
        partial_results, labels = cluster_and_partial_sums(matrix, centers, norm)
        new_centers = np.array(np.zeros(centers.shape))

        partial_results = comm.gather(partial_results, root=0)
        if rank == 0:
            # print("Partial results: %s" % partial_results)
            for partial in partial_results:
                # Mean of means, single step
                new_centers += partial / float(size)
            print("New centers: %s" % new_centers)
        comm.Bcast(new_centers, root=0)
        if np.linalg.norm(centers - new_centers, norm) < epsilon:
            print("Convergence is met")
            # Convergence criterion is met
            labels = comm.gather(labels, root=0)
            matrix = comm.gather(matrix, root=0)
            if rank == 0:
                labels = [l for sl in labels for l in sl]
                matrix = [l for sl in matrix for l in sl]

                print("Final labels: %s" % labels)
                print("Final centers: %s" % centers)
                # print("Matrix: %s" % matrix)

                plot(matrix, labels, centers)

            break
        # Convergence criterion is not met, update centers
        centers = new_centers


if __name__ == "__main__":
    main()
