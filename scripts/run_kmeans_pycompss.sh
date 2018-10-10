#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ./$(basename $0) [kmeans_args]"
    echo "E.g: ./$(basename $0) -n 1000 -c 5 -d 10 -i 6"
    exit 1
fi


cd ../src
runcompss kmeans_pycompss.py $@
