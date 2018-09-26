#!/usr/bin/env bash

appName="km_pycompss"
executable=$(pwd)/../src/kmeans_pycompss.py

if [ "$#" -gt 3 ]; then
    tracing=$1
    level=$2
    num_nodes=$3
    exe_time=$4
    shift 4
else
   echo "Usage: $(basename $0) [compss_args] [kmeans_args]"
   echo "compss_args:
        tracing log_level num_nodes exe_time"
   echo "$(python ${executable} -h) " 
   echo ""

   echo "Short demo: ./$(basename $0) true debug 2 10 -n 800 -d 5 -c 10 -i 6 -f 2"
   echo "Long demo: ./$(basename $0) true debug 4 30 -n 1677727 -d 16 -c 100 -i 6 -f 128"
   exit -1
fi

# Init sandboxed workspace and exe file

pythonpath="$(pwd)/../"

suffix=0
workspace="$(pwd)/results/${appName}_${num_nodes}_"

while [ -d "${workspace}${suffix}" ]; do
    ((++suffix))
done

mkdir -p "${workspace}${suffix}"
DEST="${workspace}${suffix}"

cd $DEST
module purge
module load intel/2017.4
module load mkl/2018.2
module load python/2.7.13
module load COMPSs/2.3


  #--qos=debug \
enqueue_compss \
  --tracing=${tracing} \
  --lang=python \
  --log_level=${level} \
  --exec_time=${exe_time} \
  --num_nodes=${num_nodes} \
  --pythonpath=/home/bsc19/bsc19277/ \
  --worker_in_master_cpus=24 \
  ${executable} $@
  
  #--worker_in_master_memory=50000 \
  #--worker_in_master_cpus=40 \
  #--cpus_per_node=100 \
  #--worker_working_dir=/lustre/home/palvarez/ \
  #--cpus_per_node=4 \
  #--cpus_per_node=4 \
  #--worker_working_dir=/lustre/home/palvarez/ \
  #--worker_working_dir=/lustre/home/palvarez/ \
  #--tracing \
  #--cpu_affinity="disabled" \
