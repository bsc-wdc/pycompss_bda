#!/usr/bin/env bash

# Parameter's check
if [ "$#" -gt 1 ]; then
    threads=$1
    timelimit=$2
    shift 2
else
   echo "Usage: $(basename $0) [mpi_args] [kmeans_args]"
   echo "mpi_args: threads max_execution_time "
   echo "$(python ../src/kmeans_mpi.py -h) "
   echo "Short demo: ./$(basename $0) 4 00:05:00 -n 800 -d 5 -c 10 -i 6"
   echo "Long demo: ./$(basename $0) 48 01:00:00 -n 1677727 -d 16 -c 100 -i 6 "
   exit -1
fi



# Manually create job for Queue System
echo "#!/bin/bash -e
#SBATCH --job-name=kmeans_mpi
#SBATCH -t${timelimit}
#SBATCH -o kmpi-%J.out
#SBATCH -e kmpi-%J.err
#SBATCH --ntasks=${threads}
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

mpirun -np ${threads} python $(pwd)/../src/kmeans_mpi.py  $@" > job

# Actual job submission to Queue System
sbatch job
