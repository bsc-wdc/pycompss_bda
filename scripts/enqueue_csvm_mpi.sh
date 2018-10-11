#!/usr/bin/env bash

# Parameter's check
if [ "$#" -ne 6 ]; then
    echo "Usage: ${scriptName} threads tracing timelimit c gamma data"
    echo "Example: ./${scriptName} 32 false 00:10:00 2 0.2 kddb"
    exit 1
fi

threads=$1
tracing=$2
timelimit=$3
threadsNode=48
c=$4
gamma=$5
data=$6
nodes=$((threads / threadsNode))



# Manually create job for Queue System
echo "#!/bin/bash -e
#SBATCH --job-name=mpicsvm
#SBATCH -t${timelimit}
#SBATCH -o log-mpicsvm-%J.out
#SBATCH -e log-mpicsvm-%J.err
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${threads}
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=${threadsNode}
#SBATCH --exclusive
#SBATCH --qos=debug


mpirun -np ${threads} python $(pwd)/../src/csvm_mpi.py train \
    -f libsvm \
    -r 5 \
    -C ${c} \
    -g ${gamma} \
    -k rbf \
    $(pwd)/../data/agaricus/train.csv" >> job

# Actual job submission to Queue System
sbatch job
