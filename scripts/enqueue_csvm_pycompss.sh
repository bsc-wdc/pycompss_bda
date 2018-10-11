#!/usr/bin/env bash

# Parameter's check
if [ "$#" -gt 2 ]; then
    num_nodes=$1
    exe_time=$2
    shift 2
else
   echo "Usage: $(basename $0) [compss_args] [kmeans_args]"
   echo "compss_args: num_nodes max_execution_time"
   echo "$(python ../src/kmeans_pycompss.py -h) "
   echo "Short demo: ./$(basename $0) 2 10 -n 800 -d 5 -c 10 -i 6 -f 96"
   echo "Long demo: ./$(basename $0) 4 60 -n 10000000 -d 50 -c 50 -i 6 -f 192"
   exit -1
fi

# Job submission to Queue System
enqueue_compss \
  --exec_time=${exe_time} \
  --num_nodes=${num_nodes} \
  --pythonpath=$(pwd)/../src/ \
  $(pwd)/../src/kmeans_pycompss.py $@
