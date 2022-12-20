#!/bin/sh

# Check that we're running this from the repo base, since we can't add
# a file from a parent directory
# https://stackoverflow.com/questions/24537340/docker-adding-a-file-from-a-parent-directory
cd "$(dirname "$0")/.."
docker run -v "$PWD:/mnt" -w /mnt --rm -it mambaorg/micromamba:1.1.0 bash -c '\
    micromamba install -y -c conda-forge --file requirements-conda.txt && micromamba env export --explicit > specfile.txt'
mv specfile.txt docker/specfile.txt
