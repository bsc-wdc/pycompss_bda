#!/usr/bin/env bash

# MPI format:
# ./enqueue_kmeans_mpi.sh tracing threads time_limit [kmeans_args]
# ./enqueue_kmeans_mpi.sh true 4 00:05:00 -n 800 -d 5 -c 10 -i 6
#
#
# pyCOMPSs format:
# ./enqueue_kmeans_pycompss.sh tracing log_level num_nodes time_limit [kmeans_args]
# ./enqueue_kmeans_pycompss.sh true debug 4 10 -n 800 -d 5 -c 10 -i 6 -f 4


# 10M points, 60 dimensions, 50 centers, 6 iterations
./enqueue_kmeans_mpi.sh false 24 00:40:00 -n 10000000 -d 60 -c 50 -i 6
./enqueue_kmeans_mpi.sh false 72 00:40:00 -n 10000000 -d 60 -c 50 -i 6
./enqueue_kmeans_mpi.sh false 168 00:40:00 -n 10000000 -d 60 -c 50 -i 6
./enqueue_kmeans_mpi.sh false 360 00:40:00 -n 10000000 -d 60 -c 50 -i 6
./enqueue_kmeans_mpi.sh false 744 00:40:00 -n 10000000 -d 60 -c 50 -i 6
./enqueue_kmeans_mpi.sh false 1512 00:40:00 -n 10000000 -d 60 -c 50 -i 6

./enqueue_kmeans_pycompss.sh false off 1 20 -n 10000000 -d 60 -c 50 -i 6 -f 24
./enqueue_kmeans_pycompss.sh false off 2 20 -n 10000000 -d 60 -c 50 -i 6 -f 72
./enqueue_kmeans_pycompss.sh false off 3 20 -n 10000000 -d 60 -c 50 -i 6 -f 168
./enqueue_kmeans_pycompss.sh false off 4 20 -n 10000000 -d 60 -c 50 -i 6 -f 360
./enqueue_kmeans_pycompss.sh false off 5 20 -n 10000000 -d 60 -c 50 -i 6 -f 744
./enqueue_kmeans_pycompss.sh false off 6 20 -n 10000000 -d 60 -c 50 -i 6 -f 1512


# 16M points, 128 dimensions, 100 centers, 6 iterations
# STRONG SCALING IS NOT SUPPORTED IN MPI (fragments are automatically are the # of threads)
#./enqueue_kmeans_mpi.sh false 24 05:40:00 -n 16000000 -d 60 -c 50 -i 6
#./enqueue_kmeans_mpi.sh false 72 04:30:00 -n 16000000 -d 60 -c 50 -i 6
#./enqueue_kmeans_mpi.sh false 168 03:20:00 -n 16000000 -d 60 -c 50 -i 6
#./enqueue_kmeans_mpi.sh false 360 02:20:00 -n 16000000 -d 60 -c 50 -i 6
#./enqueue_kmeans_mpi.sh false 744 01:00:00 -n 16000000 -d 60 -c 50 -i 6
#./enqueue_kmeans_mpi.sh false 1512 01:00:00 -n 16000000 -d 60 -c 50 -i 6
#
#./enqueue_kmeans_pycompss.sh false off 1 20 -n 10000000 -d 60 -c 50 -i 6 -f 24
#./enqueue_kmeans_pycompss.sh false off 2 20 -n 10000000 -d 60 -c 50 -i 6 -f 72
#./enqueue_kmeans_pycompss.sh false off 3 20 -n 10000000 -d 60 -c 50 -i 6 -f 168
#./enqueue_kmeans_pycompss.sh false off 4 20 -n 10000000 -d 60 -c 50 -i 6 -f 360
#./enqueue_kmeans_pycompss.sh false off 5 20 -n 10000000 -d 60 -c 50 -i 6 -f 744
#./enqueue_kmeans_pycompss.sh false off 6 20 -n 10000000 -d 60 -c 50 -i 6 -f 1512
