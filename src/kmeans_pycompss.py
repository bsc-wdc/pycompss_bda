from time import time

import numpy as np
from pycompss.api.parameter import *
from pycompss.api.task import task


def parse_arguments():
    """
    Parse command line arguments. Make the program generate
    a help message in case of wrong usage.
    """
    import argparse
    parser = argparse.ArgumentParser(description='A COMPSs Kmeans implementation.')
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
    parser.add_argument('-f', '--num_fragments', type=int)

    return parser.parse_args()


def init_board_random(numV, dim, seed):
    np.random.seed(seed)
    return np.random.random((numV, dim))


@task(returns=1)
def generate_fragment(numv, dim, seed):
    return init_board_random(numv, dim, seed)


@task(returns=1, labels=INOUT)
def cluster_and_partial_sums(fragment, labels, centres, norm):
    '''Given a fragment of points, declare a CxD matrix A and, for each point p:
    1) Compute the nearest centre c of p
    2) Add p / num_points_in_fragment to A[index(c)]
    3) Set label[index(p)] = c
    '''
    ret = np.matrix(np.zeros(centres.shape))
    n = fragment.shape[0]
    c = centres.shape[0]

    # Check if labels is an empty list
    # if not labels:
    #     # If it is, fill it with n zeros.
    #     for _ in range(n):
    #         # Done this way to not lose the reference
    #         labels.append(0)
    # Compute the big stuff
    associates = np.zeros(c)
    # Get the labels for each point
    for (i, point) in enumerate(fragment):
        distances = np.zeros(c)
        for (j, centre) in enumerate(centres):
            distances[j] = np.linalg.norm(point - centre, norm)
        labels[i] = np.argmin(distances)
        associates[labels[i]] += 1
    # Add each point to its associate centre
    for (i, point) in enumerate(fragment):
        ret[labels[i]] += point / associates[labels[i]]
    return ret


def kmeans_frag(fragments, dimensions, num_centers, max_iterations, seed, epsilon, norm):
    '''A fragment-based K-Means algorithm.
    Given a set of fragments (which can be either PSCOs or future objects that
    point to PSCOs), the desired number of clusters and the maximum number of
    iterations, compute the optimal centres and the index of the centre
    for each point.
    PSCO.mat must be a NxD float np.matrix, where D = dimensions
    '''
    import numpy as np

    # Set the random seed
    np.random.seed(seed)
    # Centres is usually a very small matrix, so it is affordable to have it in
    # the master.
    centres = np.matrix(
        [np.random.random(dimensions) for _ in range(num_centers)]
    )
    # Make a list of labels, treat it as INOUT
    # Leave it empty at the beginning, update it inside the task. Avoid
    # having a linear amount of stuff in master's memory unnecessarily
    labels = [[] for _ in range(len(fragments))]
    # Note: this implementation treats the centres as files, never as PSCOs.
    for it in range(max_iterations):
        print("Iteration: %s" % it)
        partial_results = []
        for (i, frag) in enumerate(fragments):
            # For each fragment compute, for each point, the nearest centre.
            # Return the mean sum of the coordinates assigned to each centre.
            # Note that mean = mean ( sum of sub-means )
            partial_result = cluster_and_partial_sums(frag, labels[i], centres, norm)
            partial_results.append(partial_result)
        # Bring the partial sums to the master, compute new centres when syncing
        new_centres = np.matrix(np.zeros(centres.shape))
        from pycompss.api.api import compss_wait_on
        for partial in partial_results:
            partial = compss_wait_on(partial)
            # Mean of means, single step
            new_centres += partial / float(len(fragments))
        if np.linalg.norm(centres - new_centres, norm) < epsilon:
            # Convergence criterion is met
            print("Convergence is met")
            break
        # Convergence criterion is not met, update centres
        centres = new_centres
    # If we are here either we have converged or we have run out of iterations
    # In any case, now it is time to update the labels in the master
    ret_labels = []
    for label_list in labels:
        from pycompss.api.api import compss_wait_on
        to_add = compss_wait_on(label_list)
        ret_labels += to_add
    return centres, ret_labels


'''This code is used for experimental purposes.
I.e it generates random data from some parameters that determine the size,
dimensionality and etc and returns the elapsed time.
'''


def main():
    args = parse_arguments()

    print("Execution arguments:\n%s" % args)
    t0 = time()

    # Generate the data
    fragment_list = []
    # Prevent infinite loops in case of not-so-smart users
    points_per_fragment = max(1, args.num_points // args.num_fragments)
    for l in range(0, args.num_points, points_per_fragment):
        # Note that the seed is different for each fragment. This is done to avoid
        # having repeated data.
        r = min(args.num_points, l + points_per_fragment)
        fragment_list.append(
            generate_fragment(r - l, args.dimensions, args.seed + l)
        )
    centers, labels = kmeans_frag(fragments=fragment_list,
                                  dimensions=args.dimensions,
                                  num_centers=args.num_centers,
                                  max_iterations=args.max_iterations,
                                  seed=args.seed,
                                  epsilon=args.epsilon,
                                  norm=args.lnorm)

    t1 = time()
    print("Total elapsed time: %s" % (t1 - t0))


if __name__ == "__main__":
    main()
