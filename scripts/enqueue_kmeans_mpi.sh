#!/usr/bin/env bash

appName="km_mpi"
executable=$(pwd)/../src/kmeans_mpi.py

if [ "$#" -gt 3 ]; then
    tracing=$1
    threads=$2
    timelimit=$3
    shift 3
else
   echo "Usage: $(basename $0) [compss_args] [kmeans_args]"
   echo "mpi_args:
        tracing threads time_limit "
   echo "$(python ${executable} -h) " 
   echo ""

   echo "Short demo: ./$(basename $0) true 4 00:05:00 -n 800 -d 5 -c 10 -i 6"
   echo "Long demo: ./$(basename $0) true 48 01:00:00 -n 1677727 -d 16 -c 100 -i 6 "
   exit -1
fi


threadsNode=32
nodes=$((threads / threadsNode))
baseDir=$(pwd)

# Init sandboxed workspace and exe file

suffix=0
workspace="$(pwd)/results/${appName}_${nodes}_"

while [ -d "${workspace}${suffix}" ]; do
    ((++suffix))
done

mkdir -p "${workspace}${suffix}"
DEST="${workspace}${suffix}"

cd $DEST

echo "#!/bin/bash -e
#SBATCH --job-name=${appName}
#SBATCH -t${timelimit}
#SBATCH -o kmpi-%J.out
#SBATCH -e kmpi-%J.err
#SBATCH --ntasks=${threads}
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

module purge
module load intel/2017.4
module load bsc/1.0
module load mkl/2018.2
module load impi/2018.2
module load gcc/7.2.0
module load python/2.7.13
" > job

if [ "$tracing" == "true" ]; then
  echo "mpirun -np ${threads} ${baseDir}/trace.sh \\" >> job
else
  echo "mpirun -np ${threads} \\" >> job
fi

echo "python ${executable} $@" >> job

sbatch job
