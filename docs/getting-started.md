## Install


`dolphin` is available on conda-forge:

```bash
mamba install -c conda-forge dolphin
```

`dolphin` has the ability to unwrap interferograms using `isce3`'s python bindings to [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/).
It is also integrated with [`tophu`](https://github.com/isce-framework/tophu) to unwrap large interferograms in parallel tiles at multiple resolution.
To install both dolphin and tophu through conda-forge, run
```bash
mamba install -c conda-forge tophu dolphin
```

## Usage

The main entry point for running the phase estimation/stitching and unwrapping workflows is named `dolphin`, which has two subcommands:

1. `dolphin config`: create a workflow configuration file.
2. `dolphin run` : run the workflow using this file.

Example usage:

```bash
$ dolphin config --slc-files /path/to/slcs/*tif
```
This will create a YAML file (by default `dolphin_config.yaml` in the current directory).

The only required inputs for the workflow is a list of coregistered SLC files (in either geographic or radar coordinates).
If the SLC files are spread over multiple files, you can either
1. use the `--slc-files` option with a bash glob pattern:

```bash
dolphin config --slc-files /path/to/SLCs/*/*.slc
```
Another example: `dolphin config --slc-files merged/SLC/*/*.slc` would match the [ISCE2 stack processor output](https://github.com/isce-framework/isce2/tree/main/contrib/stack) )


2. Store all input SLC files in a text file delimited by newlines (e.g. `my_slc_list.txt`), and give the name of this text file prefixed by the `@` character :

```bash
dolphin config --slc-files @my_slc_list.txt
```

The full set of options is written to the configuration file; you can edit this file, or you can see which commonly tuned options by are changeable running `dolphin config --help`.


## Setup for Developers

To contribute to the development of `dolphin`, you can fork the repository and install the package in development mode.
We encourage new features to be developed on a new branch of your fork, and then submitted as a pull request to the main repository.

To install locally,

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
mamba activate dolphin-env
python -m pip install -e .
```


If you have access to a GPU, you can install the extra requirements from running the GPU accelerated algorithms:
```bash
mamba env update --name dolphin-env --file conda-env-gpu-extras.yml
```


The extra packages required for testing and building the documentation can be installed:
```bash
# Run "pip install -e" to install with extra development requirements
python -m pip install -e ".[docs,test]"
```

We use [`pre-commit`](https://pre-commit.com/) to automatically run linting and formatting:
```bash
# Get pre-commit hooks so that linting/formatting is done automatically
pre-commit install
```
This will set up the linters and formatters to run on any staged files before you commit them.

After making functional changes, you can rerun the existing tests and any new ones you have added using:
```bash
python -m pytest
```

### GPU setup
If you have access to a GPU, you can install the extra requirements from running the GPU accelerated algorithms:

```bash
mamba env update -n dolphin-env --file conda-env-gpu-extras.yml
```
Note that the version of `cudatoolkit` must match the drivers installed for your GPU (which may come from the output of `nvidia-smi`)
See the [numba](https://numba.readthedocs.io/en/stable/cuda/overview.html#software) and [cupy](https://docs.cupy.dev/en/stable/install.html) installation instructions for more details on getting set up.

To check whether you have successfully installed `numba` and `cupy`, run
```bash
python -c 'from dolphin import utils; print(utils.gpu_available())'
```

### Creating Documentation


We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](mkdocstrings.github.io/).

When adding new documentation, you can build and serve the documentation locally using:

```
mkdocs serve
```
then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.
