#!/usr/bin/env bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

readonly USAGE="usage: $0 [-t TAG] [-u MAMBA_USER_ID] [-b BASE]"
readonly HELP="$USAGE

Build the docker image for dolphin.

options:
-t, --tag TAG           Specify a name/tag for the docker image. Default: isce-framework/dolphin:latest
-u, --user-id USER_ID   Specify the user id to use in the docker image. Default: 1000
-b, --base BASE         Specify the base image to use. Default: ubuntu:22.04
-h, --help              Show this help message and exit
"

# Parse input arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -t | --tag)
        TAG="$2"
        shift
        shift
        ;;
        # Add the ability to specify build-arg MAMBA_USER_ID
    -u | --user-id)
        MAMBA_USER_ID="$2"
        shift
        shift
        ;;
        # Add ability to specify base image
    -b | --base)
        BASE="$2"
        shift
        shift
        ;;
    -h | --help)
        echo "$HELP"
        exit 0
        ;;
    *)
        echo "Unknown option: $key"
        echo "$USAGE"
        exit 1
        ;;
    esac
done

image_name=${TAG:-"isce-framework/dolphin:latest"}

# Build the Docker image
cmd_base="docker build --network=host --tag $image_name --file docker/Dockerfile"

# append --build-arg if specified:
if [ -z "${BASE+x}" ]; then
    cmd_base="$cmd_base"
else
    cmd_base="$cmd_base --build-arg BASE=$BASE"
fi

# append MAMBA_USER_ID if specified:
if [ -z "${MAMBA_USER_ID+x}" ]; then
    cmd_base="$cmd_base"
else
    cmd_base="$cmd_base --build-arg MAMBA_USER_ID=$MAMBA_USER_ID"
fi

# finish with ".":
cmd_base="$cmd_base ."
echo $cmd_base
# Run the command
eval $cmd_base

# To run the image and see the help message....
echo "To run the image and see the help message:"
echo "docker run --rm -it $image_name dolphin --help"
#
echo "To run on a workflow config file:"
echo "docker run --user \$(id -u):\$(id -g) -v \$PWD:/work --rm -it $image_name dolphin run dolphin_config.yaml"
# where...
#     --user $(id -u):$(id -g)  # Needed to avoid permission issues when writing to the mounted volume.
#     -v $PWD:/work  # Mounts the current directory to the /work directory in the container.
#     --rm  # Removes the container after it exits.
#     -it  # Needed to keep the container running after the command exits.
#     isce-framework/dolphin:latest  # The name of the image to run.
#     dolphin run dolphin_config.yaml # The `dolphin` command that is run in the container
