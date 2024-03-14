
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

## Creating Documentation

We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](mkdocstrings.github.io/).

When adding new documentation, you can build and serve the documentation locally using:

```
mkdocs serve
```
then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.

For citations, use the notation `[@Ansari2018EfficientPhaseEstimation]` to refer to a Bibtex key in `docs/references.bib`  (e.g. [@Ansari2018EfficientPhaseEstimation]).
This can be done in either a markdown file, or in a docstring.
