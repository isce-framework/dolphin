#!/bin/sh
docker run -v "$PWD:/mnt" -w /mnt --rm -it mambaorg/micromamba:1.1.0 bash -c '\
    micromamba install -y -c conda-forge --file requirements-conda.txt && micromamba env export --explicit > specfile.txt'
