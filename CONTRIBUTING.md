# Contributing

If you think you've found a problem, please let us know! You can raise an [issue](https://github.com/isce-framework/dolphin/issues) on the repository, where there are templates for Bug Reports and Feature Requests.

For more general Q&A, please use the [Discussions](https://github.com/isce-framework/dolphin/discussions) page.

If you want to make changes or add to `dolphin`, you can follow the development setup:

## Development Installation

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

## Setting up your coding environment

We use [`pre-commit`](https://pre-commit.com/) to automatically run linting, formatting, and the [`mypy`](https://mypy.readthedocs.io/en/stable/) static type checker.

```bash
# Get pre-commit hooks so that linting/formatting is done automatically
pre-commit install
```

This will set up the linters and formatters to run on any staged files before you commit them.

It is recommended to install [`ruff`](https://docs.astral.sh/ruff/) into your editor so that the linting/formatting problems will be evident to you before you try to commit.

## Running tests

After making functional changes, you can rerun the existing tests using [`pytest`](https://docs.pytest.org).
The extra packages required for testing can be installed:

```bash
# Run "pip install -e" to install with extra development requirements
python -m pip install -e ".[test]"
```

```bash
python -m pytest
```

For any new functionality, we ask that you write new unit tests in the module.
For bug fixes, a good practice is to write a test which fails with the current code, then add the fix and ensure the test passes.

## Creating Documentation

We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](https://mkdocstrings.github.io/python/)

The dependencies for building and viewing the documentation locally can be installed:

```bash
python -m pip install -e ".[docs]"
```


When adding new documentation, you can build and serve the documentation locally using:

```bash
mkdocs serve
```

then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.

For citations, use the notation `[@Ansari2018EfficientPhaseEstimation]` to refer to a Bibtex key in `docs/references.bib`  (e.g. [@Ansari2018EfficientPhaseEstimation]).
This can be done in either a markdown file, or in a docstring.
