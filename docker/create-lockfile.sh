#!/usr/bin/env bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

readonly HELP='usage: ./create-lockfile.sh ENVFILE > specfile.txt

Create a conda lockfile from an environment YAML file for reproducible
environments.

positional arguments:
ENVFILE     a YAML file containing package specifications

options:
-h, --help  show this help message and exit
'

main() {
    # Get absolute path of input YAML file.
    local ENVFILE
    ENVFILE=$(realpath "$1")

    # Get concretized package list.
    local PKGLIST
    PKGLIST=$(docker run --network=host \
        -v "$ENVFILE:/tmp/environment.yml:ro" --rm \
        mambaorg/micromamba:1.1.0 bash -c '\
            micromamba install -y -n base -f /tmp/environment.yml > /dev/null && \
            micromamba env export --explicit')

    # Sort packages alphabetically.
    # (The first 4 lines are assumed to be header lines and ignored.)
    echo "$PKGLIST" | (sed -u 4q; sort)
}

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo "$HELP"
elif [[ "$#" -ne 1 ]]; then
    echo 'Illegal number of parameters' >&2
    echo "$HELP"
    exit 1
else
    main "$@"
fi
