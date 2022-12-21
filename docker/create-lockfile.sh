#!/bin/sh
set -e

# Check that we're running this from the repo base, since we can't add
# a file from a parent directory
# https://stackoverflow.com/questions/24537340/docker-adding-a-file-from-a-parent-directory
cd "$(dirname "$0")/.."
docker run -u $(id -u):$(id -g) -v "$PWD:/mnt" -w /mnt --rm -it mambaorg/micromamba:1.1.0 bash -c '\
    micromamba create --yes --name dolphin-env --file conda-env.yml && micromamba env export --name dolphin-env --explicit > /mnt/specfile.txt'
mv specfile.txt docker/specfile.txt
