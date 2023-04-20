#!/bin/bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

USAGE="Usage: $0 -b|--benchmark-dir [-i|--image <docker image>]"

# Parse input arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -i | --image)
        IMAGE="$2"
        shift
        shift
        ;;
    -b | --benchmark-dir)
        BENCH_DIR="$2"
        shift
        shift
        ;;
    -h | --help)
        echo "$USAGE"
        exit 0
        ;;
    *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

if [ -z "${BENCH_DIR+x}" ]; then
    echo "Please specify a benchmark directory."
    echo "$USAGE"
    exit 1
fi

if [ -z "${IMAGE+x}" ]; then
    IMAGE="opera-adt/dolphin:latest"
    echo "Using default docker image: $IMAGE"
fi

# Run the SAS workflow.
DATA_DIR="$BENCH_DIR/data"

# Check that the data directory exists.
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory does not exist: $DATA_DIR"
    exit 1
fi
# Check we have input_slcs/ and dynamic_ancillary/ config_files/ and
# config_files/dolphin_config.yaml
if [ ! -d "$DATA_DIR/input_slcs" ]; then
    echo "Input SLCs directory does not exist: $DATA_DIR/input_slcs"
    exit 1
fi
if [ ! -d "$DATA_DIR/dynamic_ancillary" ]; then
    echo "Dynamic ancillary directory does not exist: $DATA_DIR/dynamic_ancillary"
    exit 1
fi
if [ ! -d "$DATA_DIR/config_files" ]; then
    echo "Config files directory does not exist: $DATA_DIR/config_files"
    exit 1
fi

cd $DATA_DIR
WORKDIR="/tmp"
mkdir -p scratch output
rm -rf scratch/* output/*
docker run --rm --user=$(id -u):$(id -g) \
    --volume="$(realpath input_slcs):$WORKDIR/input_slcs:ro" \
    --volume="$(realpath dynamic_ancillary):$WORKDIR/dynamic_ancillary:ro" \
    --volume="$(realpath config_files/dolphin_config.yaml):$WORKDIR/dolphin_config.yaml:ro" \
    --volume="$(realpath scratch):$WORKDIR/scratch" \
    --volume="$(realpath output):$WORKDIR/output" \
    --workdir="$WORKDIR" \
    "$IMAGE" dolphin run dolphin_config.yaml

LOG_DIR="$BENCH_DIR/logs"
mkdir -p $LOG_DIR

# TODO: run this in a docker container.
eval "$(conda shell.bash hook)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mapping
python /u/aurora-r0/staniewi/repos/dolphin/scripts/benchmarking/parse_benchmark_logs.py \
    --config-files config_files/dolphin_config.yaml \
    --outfile "$LOG_DIR/benchmark_results_$(date +%Y%m%d).csv"
