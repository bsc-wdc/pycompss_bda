# PyCOMPSs Big Data Analytics


Available applications in the repository:

* Cascade Support Vector Machines
* K-means

The applications are implemented using both PyCOMPSs and K-means for comparison purposes.

To know more check:

* [PyCOMPSs](https://github.com/bsc-wdc/compss)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

## Running the Apps

The folder `./scripts` contains samples to run the for implementaions.

The scripts named `run_APP_IMPLEMENTATION.sh` are used to run the applications locally.

The scripts named `enqueue_APP_IMPLEMENTATION.sh` are used to run the applications in a supercomputer.

In the case of PyCOMPSs version, the script can be used out-of-the-gox in many supercomputers. The MPI version is tied to SLURM Queueing systems because MPI is not platform-agnostic as PyCOMPSs.

## Computing Complexity

In order to compute the complexity metrics of the K-means and C-SVM applications for both MPI and PyCOMPSs run the script:

`./scripts/get_complexities.sh`

This script reports 3 complexity metrics:

* Source Lines of Code (SLOC)
* Cyclomatic complexity
* NPath complexity

**Requirements**:

This script uses Babelfish tools to compute the Cyclomatic and NPath complexities, and cloc for the SLOC.

* [Babelfish Tools](https://github.com/bblfsh/tools)
* [Cloc](https://github.com/AlDanial/cloc)
