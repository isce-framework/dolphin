## COMPASS

COregistered Multi-temPorAl Sar Slc

🚨 This toolbox is still in **pre-alpha** stage and undergoing **rapid development**. 🚨

### Install

The following will install COMPASS into a conda environment.

1. Download source code:

```bash
git clone https://github.com/opera-adt/COMPASS.git
```

2. Install dependencies:

```bash
conda install -c conda-forge --file COMPASS/requirements.txt
python -m pip install git+https://github.com/opera-adt/s1-reader.git
```

3. Install `COMPASS` via pip:

```bash
# run "pip install -e" to install in development mode
python -m pip install ./COMPASS
```

### Usage

The following commands generate coregistered SLC in radar or geo-coordinates from terminal:

```bash
s1_cslc.py --grid geo   <path to s1_cslc_geo   yaml file>
s1_cslc.py --grid radar <path to s1_cslc_radar yaml file for reference burst>
s1_cslc.py --grid radar <path to s1_cslc_radar yaml file for secondary burst>
```

### License
**Copyright (c) 2021** California Institute of Technology (“Caltech”). U.S. Government
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
