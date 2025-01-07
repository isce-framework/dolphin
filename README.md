# dolphin

[![Pytest and build docker image](https://github.com/isce-framework/dolphin/actions/workflows/test-build-push.yml/badge.svg?branch=main)](https://github.com/isce-framework/dolphin/actions/workflows/test-build-push.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isce-framework/dolphin/main.svg)](https://results.pre-commit.ci/latest/github/isce-framework/dolphin/main)
[![Documentation Status][rtd-badge]][rtd-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06997/status.svg)](https://doi.org/10.21105/joss.06997)

<!-- prettier-ignore-start -->
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/dolphin
[conda-link]:               https://github.com/conda-forge/dolphin-feedstock
[pypi-link]:                https://pypi.org/project/dolphin/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dolphin
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/isce-framework/dolphin/discussions
[rtd-badge]:                https://readthedocs.org/projects/dolphin-insar/badge/?version=latest
[rtd-link]:                 https://dolphin-insar.readthedocs.io/en/latest/?badge=latest
<!-- prettier-ignore-end -->

High resolution wrapped phase estimation for Interferometric Synthetic Aperture Radar (InSAR) using combined persistent scatterer (PS) and distributed scatterer (DS) processing.

<!-- DeformatiOn Land surface Products in High resolution using INsar -->

## Install

`dolphin` may be installed via conda-forge:

```bash
# if mamba is not already installed, see here: https://mamba.readthedocs.io/en/latest/
mamba install -c conda-forge dolphin
```

It is also available via [`PyPI`](https://pypi.org/project/dolphin/) and may be `pip`-installed on some platforms, such as Google's Colab. However, certain dependencies (e.g. GDAL) are more easily set up through `conda`.

`dolphin` has the ability to unwrap interferograms using several options, which can be toggled using the `unwrap_method` configuration option:

1. [`snaphu-py`](https://github.com/isce-framework/snaphu-py), a lightweight Python bindings to [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/). Available on both pip and conda.
2. [`isce3`](https://github.com/isce-framework/isce3)'s python bindings to SNAPHU, PHASS, or ICU
3. [`spurt`](https://github.com/isce-framework/spurt), a 3D unwrapper, implementing the Extended Minimum Cost Flow (ECMF) algorithm
4. [`tophu`](https://github.com/isce-framework/tophu), a multi-scale unwrapper designed to unwrap large interferograms in parallel tiles at multiple resolution.


These may be installed via conda or (in the case of `snaphu-py`) pip.

To install locally:

1. Download source code:

```bash
git clone https://github.com/isce-framework/dolphin.git && cd dolphin
```

2. Install dependencies:

```bash
mamba env create --file conda-env.yml
```

or if you have an existing environment:

```bash
mamba env update --name my-existing-env --file conda-env.yml
```

3. Install `dolphin` via pip:

```bash
conda activate dolphin-env
python -m pip install .
```

Dolphin can also take advantage of CUDA-compatible GPUs for faster processing. [See the docs](https://dolphin-insar.readthedocs.io/en/latest/gpu-setup) for installation instructions and configuration.

## Usage

The main entry point for configuring and running workflows the `dolphin` command line tool:

1. `dolphin config`: create a workflow configuration file.
2. `dolphin run` : run the workflow using this file.

Example usage:

```bash
dolphin config --slc-files /path/to/slcs/*tif
# OR: to make a coarser output 4x as quickly:
# dolphin config --slc-files /path/to/slcs/*tif --strides 2 2
dolphin run dolphin_config.yaml
```

The `config` command creates a YAML file (by default `dolphin_config.yaml` in the current directory). If you'd like to see an empty YAML with all defaults filled in, you can run `dolphin config --print-empty`, which creates a [sample file like the one here](https://raw.githubusercontent.com/isce-framework/dolphin/refs/heads/main/docs/sample_dolphin_config.yaml)

The only required inputs for the workflow are the paths to the coregistered SLC files (in either geographic or radar coordinates).
If the SLC files are spread over multiple files, you can either

1. use the `--slc-files` option with a bash glob pattern, (e.g. `dolphin config --slc-files merged/SLC/*/*.slc` would match the [ISCE2 stack processor output](https://github.com/isce-framework/isce2/tree/main/contrib/stack) )

1. Store all input SLC files in a text file delimited by newlines (e.g. `my_slc_list.txt`), and give the name of this text file prefixed by the `@` character (e.g. `dolphin config --slc-files @my_slc_list.txt`)

The full set of options is written to the configuration file; you can edit this file, or you can see which commonly tuned options by are changeable running `dolphin config --help`.

## Building and running via Docker

`dolphin` can also be run using Docker. You can use the one built on [Github](https://github.com/isce-framework/dolphin/pkgs/container/dolphin), or build it locally using the script

```bash
./docker/build-docker-image.sh
```

## Contributing

We welcome many forms of contributing, including testing, bug reports, and documentation fixes. If you think you've found a problem, please let us know! You can raise an [issue](https://github.com/isce-framework/dolphin/issues) on the repository, where there are templates for Bug Reports and Feature Requests. If you have a general question of idea, feel free to raise it in the [Discussions](https://github.com/isce-framework/dolphin/discussions) page.

For more detailed guidance on setting up a development environment, including how make and test changes to the code, see [Contributing to Dolphin](CONTRIBUTING.md).

For more general Q&A, please use the [Discussions](https://github.com/isce-framework/dolphin/discussions) page.

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
