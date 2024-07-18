#!/usr/bin/env bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

readonly HELP='usage: ./create-lockfile.sh --file ENVFILE [--pkgs PACKAGE ...] > specfile.txt

Create a conda lockfile from an environment YAML file and additional packages for reproducible environments.

options:
--file ENVFILE    Specify a YAML file containing package specifications.
--pkgs PACKAGE    Specify additional packages separated by spaces. Example: --pkgs numpy scipy
-h, --help        Show this help message and exit
'

install_packages() {
    local ENVFILE=$(realpath "$1")
    shift
    local PACKAGES="$@"

    # Prepare arguments for the command
    local FILE_ARG="--file /tmp/$(basename "$ENVFILE")"
    if [[ -n "$PACKAGES" ]]; then
        PKGS_ARGS=(${PACKAGES[@]})
    else
        PKGS_ARGS=""
    fi

    # Get concretized package list.
    local PKGLIST
    PKGLIST=$(docker run --rm --network=host \
        -v "$ENVFILE:/tmp/$(basename "$ENVFILE"):ro" \
        mambaorg/micromamba:1.1.0 bash -c "\
            micromamba install -y -n base $FILE_ARG $PKGS_ARGS > /dev/null && \
            micromamba env export --explicit")

    # Sort packages alphabetically.
    # (The first 4 lines are assumed to be header lines and ignored.)
    echo "$PKGLIST" | (
        sed -u 4q
        sort
    )
}

main() {
    local ENVFILE=""
    local PACKAGES=()

    while [[ "$#" -gt 0 ]]; do
        case $1 in
        --file)
            shift
            if [[ -z "${1-}" ]]; then
                echo "No file provided after --file" >&2
                exit 1
            fi
            ENVFILE="$1"
            shift
            ;;
        --pkgs)
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                PACKAGES+=("$1")
                shift
            done
            ;;
        -h | --help)
            echo "$HELP"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "$HELP"
            exit 1
            ;;
        esac
    done

    if [[ -z "$ENVFILE" ]]; then
        echo 'No environment file provided' >&2
        echo "$HELP"
        exit 1
    fi

    # If no packages were passed, install only the packages in the environment file.
    if [[ "${#PACKAGES[@]}" -eq 0 ]]; then
        install_packages "$ENVFILE"
    else
        install_packages "$ENVFILE" "${PACKAGES[@]}"
    fi
}

main "$@"
