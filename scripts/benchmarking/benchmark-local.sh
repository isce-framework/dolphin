#!/bin/bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

# # Build the docker image.
# BASE="ubuntu:20.04"
# TAG="opera-adt/dolphin:ubuntu-latest"
# docker build --network=host \
#     --build-arg="BASE=$BASE" \
#     --tag="$TAG" \
#     --file docker/Dockerfile .

# Pull from ghcr.io
# TODO!

# Run the SAS workflow.
DATA_DIR="/home/staniewi/dev/dolphin-benchmarks/data"
cd $DATA_DIR

WORKDIR="/tmp"
mkdir -p scratch
mkdir -p output
docker run --rm --user=$(id -u):$(id -g) \
    --volume="$(realpath input_slcs):$WORKDIR/input_slcs:ro" \
    --volume="$(realpath dynamic_ancillary):$WORKDIR/dynamic_ancillary:ro" \
    --volume="$(realpath config_files/dolphin_config.yaml):$WORKDIR/dolphin_config.yaml:ro" \
    --volume="$(realpath scratch):$WORKDIR/scratch" \
    --volume="$(realpath output):$WORKDIR/output" \
    --workdir="$WORKDIR" \
    "$TAG" dolphin run dolphin_config.yaml

# dolphin_config_cpu_block1GB_strides2_tpw8_nslc27_nworkers4.log

cp output/dolphin.log
eval "$(conda shell.bash hook)"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mapping

python /u/aurora/staniewi/repos/dolphin/dolphin-benchmarks/scripts/plot.py

# # Compare the output against a golden dataset.
# docker run --rm --user=$(id -u):$(id -g) \
#     --volume="$(realpath golden_output):$WORKDIR/golden_output:ro" \
#     --volume="$(realpath output):$WORKDIR/output:ro" \
#     --workdir="$WORKDIR" \
#     "$TAG" \
#     /dolphin/scripts/release/validate_product.py golden_output/20180101_20180716.unw.nc output/20180101_20180716.unw.nc
