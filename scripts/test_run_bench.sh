#!/bin/bash
set -e

# Save the commits hashes for all tagged releases
# Only look for ones past version 0.4.1 (which added `evd` option)
git show-ref --tags | awk '$2 >= "refs/tags/v0.4.2" { print $1; } ' >tag_hashlist.txt

num_threads=${NUM_THREADS:-16}
export NUMBA_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
asv run HASHFILE:tag_hashlist.txt --skip-existing-commits \
    --cpu-affinity 0-15 \
    --show-stderr
