# Unreleased, 0.0

**Added**

- Support for upsampling N-D arrays (FFT-based & nearest neighbor)
- Basic support for multilooking
- Band pass FIR filter implementation using the optimal equiripple method
- Tile manager class
- Abstract interface to "plug-in" unwrapping algorithms
- Unwrapping via SNAPHU, PHASS, and ICU
- Baseline multi-scale unwrapping implementation

**Changed**

**Deprecated**

**Removed**

**Fixed**

**Dependencies**

Added requirements:

- gdal>=3.0
- numpy
- h5py
- ruamel.yaml
- yamale
- numba
- pip:
  - pymp-pypi
