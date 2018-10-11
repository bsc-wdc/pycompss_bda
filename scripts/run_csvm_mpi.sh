#!/usr/bin/env bash

# Parameter's check
if [ "$#" -lt 1 ]; then
    echo "Usage: ./$(basename $0) threads [csvm_args]"
    echo "csvm_args: $(python ../src/csvm_mpi.py -h) "
    echo "Short demo: ./$(basename $0) 4"
    exit 1
fi

threads=$1

mpirun -np ${threads} python $(pwd)/../src/csvm_mpi.py train \
    -f csv \
    -r 10 \
    -C 10000 \
    -g 0.01 \
    -k rbf \
    -s model \
    $(pwd)/../data/agaricus/train.csv
