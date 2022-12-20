# dolphin

Algorithms and Tools for LAnd Surface Deformation using InSAR

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨



## Usage

Dolphin has a main command line entry point to run the algorithms and tools in workflows.
The main entry point is named `dolphin`, which has two subcommands:

1. `dolphin config`: create a workflow configuration file.
2. `dolphin run` : run the workflow using this file.

Example usage:

```bash
$ dolphin config --slc-directory /path/to/slc --ext ".tif"
```
This will create a YAML file (by default `dolphin_config.yaml` in the current directory).
You can also directly use a list of SLC files as input, e.g.:
```bash
$ dolphin config --slc-files /path/to/slc1.tif /path/to/slc2.tif
```



## Install

The following will install `dolphin` into a conda environment.

1. Download source code:
```bash
git clone https://github.com/opera-adt/dolphin.git && cd dolphin
```
2. Install dependencies:
```bash
# assuming that mamba is installed: https://mamba.readthedocs.io/en/latest/
# if not, start with:
# conda install mamba -n base -c conda-forge
mamba install -c conda-forge --file requirements.txt
```
3. Install `dolphin` via pip:
```bash
# -e installs in development mode
python -m pip install -e .
```

For development:

```bash
# run "pip install -e" to install with extra development requirements
python -m pip install -e .[docs]
# Get pre-commit hooks so that linting/formatting is done automatically
pre-commit install

# After making changes, check the tests:
pytest
```

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
