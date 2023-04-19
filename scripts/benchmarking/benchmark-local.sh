#!/bin/bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

# Parse input arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -t | --tag)
        TAG="$2"
        shift
        shift
        ;;
    *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

if [ -z "${TAG+x}" ]; then
    TAG="opera-adt/dolphin:latest"
    echo "Using default tag: $TAG"
fi

# Run the SAS workflow.
BENCH_DIR="/u/aurora-r0/staniewi/dev/dolphin-benchmarks/"
DATA_DIR="$BENCH_DIR/data"
cd $DATA_DIR

# WORKDIR="/tmp"
mkdir -p scratch output
rm -rf scratch/* output/*
docker run --rm --user=$(id -u):$(id -g) \
    --volume="$(realpath input_slcs):$WORKDIR/input_slcs:ro" \
    --volume="$(realpath dynamic_ancillary):$WORKDIR/dynamic_ancillary:ro" \
    --volume="$(realpath config_files/dolphin_config.yaml):$WORKDIR/dolphin_config.yaml:ro" \
    --volume="$(realpath scratch):$WORKDIR/scratch" \
    --volume="$(realpath output):$WORKDIR/output" \
    --workdir="$WORKDIR" \
    "$TAG" dolphin run dolphin_config.yaml

dolphin run config_files/dolphin_config.yaml

eval "$(conda shell.bash hook)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mapping

LOG_DIR="$BENCH_DIR/logs"
mkdir -p $LOG_DIR
python /u/aurora-r0/staniewi/repos/dolphin/scripts/benchmarking/parse_benchmark_logs.py \
    --config-files output/dolphin.log \
    --outfile "$LOG_DIR/benchmark_results_$(date +%Y%m%d).csv"
