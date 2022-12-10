
--8<--
README.md:usage
--8<--

--8<--
README.md:install
--8<--


## Creating Documentation


We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](mkdocstrings.github.io/).

When adding new documentation, you can build and serve the documentation locally using:

```
mkdocs serve
```
then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.


The online documentation is hosted using Github Pages and versioned using [Mike](https://github.com/jimporter/mike/issues).


### Manually deploying new versions

(copied from https://github.com/squidfunk/mkdocs-material-example-versioning)

Make a change to docs/index.md, and publish the first version:

```
mike deploy --push --update-aliases 0.1 latest
```
Set the default version to latest

```
mike set-default --push latest
```
