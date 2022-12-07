# dolphin

Algorithms and Tools for LAnd Surface Deformation using InSAR

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨


<!-- This is for snippets to copy sections into other documentation pages: -->
<!-- https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippet-sections -->
# --8<-- [start:install]
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
# --8<-- [end:install]


## Usage

## License
**Copyright (c) 2022** California Institute of Technology (“Caltech”). U.S. Government
sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
