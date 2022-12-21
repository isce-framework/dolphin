# dolphin

Algorithms and Tools for LAnd Surface Deformation using InSAR

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨



## Usage

The main entry point for running the phase estimation workflow is named `dolphin`, which has two subcommands:

1. `dolphin config`: create a workflow configuration file.
2. `dolphin run` : run the workflow using this file.

Example usage:

```bash
$ dolphin config --slc-files /path/to/slcs/*tif
```
This will create a YAML file (by default `dolphin_config.yaml` in the current directory).
You can also directly use a list of SLC files as input, e.g.:
```bash
$ dolphin config --slc-files /path/to/slc1.tif /path/to/slc2.tif
```

The only required input for the workflow is a list of coregistered SLC files (in either geographic or radar coordinates).
If the SLC files are spread over multiple files, you can either
1. use the `--slc-files` option with a bash glob pattern, (e.g. `dolphin config --slc-files merged/SLC/*/*.slc` would match the [ISCE2 stack processor output](https://github.com/isce-framework/isce2/tree/main/contrib/stack) )
1. Store all input SLC files in a text file delimited by newlines (e.g. `my_slc_list.txt`), and give the name of this text file prefixed by the `@` character (e.g. `dolphin config --slc-files @my_slc_list.txt`)


## Install

The following will install `dolphin` into a conda environment.

1. Download source code:
```bash
git clone https://github.com/opera-adt/dolphin.git && cd dolphin
```
2. Install dependencies:
```bash
conda install -c conda-forge --file conda-env.yml
```

3. Install `dolphin` via pip:
```bash
python -m pip install .
```

See the [documentation](https://dolphin-insar.readthedocs.io/) for more details.

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
