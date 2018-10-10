#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ./$(basename $0) threads [kmeans_args]"
    echo "E.g: ./$(basename $0) 4 -n 1000 -c 5 -d 10 -i 6"
    exit 1
fi

threads=$1

shift 1

mpirun -np ${threads} \
    python ../src/kmeans_mpi.py $@
