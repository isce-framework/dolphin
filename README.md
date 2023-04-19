# dolphin
[![Pytest and build docker image](https://github.com/opera-adt/dolphin/actions/workflows/test-build-push.yml/badge.svg?branch=main)](https://github.com/opera-adt/dolphin/actions/workflows/test-build-push.yml)

High resolution wrapped phase estimation for InSAR using combined PS/DS processing.

<!-- DeformatiOn Land surface Products in High resolution using INsar -->



## Install

The following will install `dolphin` into a conda environment.

1. Download source code:
```bash
git clone https://github.com/opera-adt/dolphin.git && cd dolphin
```
2. Install dependencies:
```bash
conda env create --file conda-env.yml
```

or if you have an existing environment:
```bash
conda env update --name my-existing-env --file conda-env.yml
```

3. Install `dolphin` via pip:
```bash
conda activate dolphin-env
python -m pip install .
```


## Usage

The main entry point for running the phase estimation workflow is named `dolphin`, which has two subcommands:

1. `dolphin config`: create a workflow configuration file.
2. `dolphin run` : run the workflow using this file.

Example usage:

```bash
$ dolphin config --slc-files /path/to/slcs/*tif
$ dolphin run dolphin_config.yaml
```
The `config` command creates a YAML file (by default `dolphin_config.yaml` in the current directory).

The only required inputs for the workflow is a list of coregistered SLC files (in either geographic or radar coordinates).
If the SLC files are spread over multiple files, you can either
1. use the `--slc-files` option with a bash glob pattern, (e.g. `dolphin config --slc-files merged/SLC/*/*.slc` would match the [ISCE2 stack processor output](https://github.com/isce-framework/isce2/tree/main/contrib/stack) )
1. Store all input SLC files in a text file delimited by newlines (e.g. `my_slc_list.txt`), and give the name of this text file prefixed by the `@` character (e.g. `dolphin config --slc-files @my_slc_list.txt`)

The full set of options is written to the configuration file; you can edit this file, or you can see which commonly tuned options by are changeable running `dolphin config --help`.

See the [documentation](https://dolphin-insar.readthedocs.io/) for more details.

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
