#!/usr/bin/env bash

# Run locally sample execution with agaricus dataset
cd ../src

runcompss csvm_pycompss.py \
        -i 2 \
        --convergence \
        $(pwd)/../data/sample
