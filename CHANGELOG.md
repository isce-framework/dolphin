# Unreleased

**Added**

- Created first version of the single-update workflow, usable with `dolphin config --single`
- `_background.py` module as the abstract classes for background input/output with `EagerLoader` and `Writer`.
- `from_vrt_file` for the `VRTInterferogram` class.
- Arbitrary interferogram index selection in `Network` class.
- Parallel CPU eigenvector finding using `scipy.linalg.eigh`.
- PS selection for strided outputs using the average PS phase within a window (that contains multiple PS).
- Comments in the YAML file output by the `dolphin config` command.


**Changed**

- The main workflow has been renamed to `s1_disp.py` to better reflect the workflow, since it can handle both single and stack workflows.
    - The `sequential.py` and `single.py` are where these differences are handled.
- More uniform naming in `io.get_raster_<X>` functions.
- The SLC compression is now done in `_compress.py` to declutter the `mle.py` module.

- Docs are now using the mkdocs `material` theme.

**Removed**

- `utils.parse_slc_strings` in favor of always using `utils.get_dates`.
- `io.get_stack_nodata_mask`. This will be done using the nodata polygon, or not at all.


**Dependencies**

Added requirements:

- pyproj>=3.2
- threadpoolctl>=3.0

For docs:
- mkdocs-material
- pymdown-extensions


# 0.0.3

**Added**

- Ability for `VRTStack` to handle HDF5 files with subdatasets.
    - The OPERA specific HDF5 files are now supported without extra configuration.
- First stitching of interferograms in `stitching.py`.
    - Users can pass multiple SLC burst (like COMPASS bursts) per date, and the workflow will process per stack then stitch per date.
- More features for `load_gdal` to load in blocks.

**Changed**

- A small amount of regularization on the coherence matrix is done before inversion during phase linking to avoid singular matrices.
- Renamed module to `_log.py`
- `workflows/wrapped_phase.py` absorbed much logic formerly in `s1_disp_stack.py`.

# 0.0.2

**Added**

- Created first version of the `s1_disp_stack.py` workflow.
- Created the modules necessary for first version of the sequential workflow, including
    - `ps.py`
    - `sequential.py`
    - `io.py`
    - `interferogram.py`
    - `utils.py`
    - `unwrap.py`
    - `stitching.py`
    - `vrt.py`
- Created the `phase_link` subpackage for wrapped phase estimation.


Added requirements:

- pyproj>=3.2
- tqdm>=4.60


# 0.0.1

**Added**

- Created the `config` module to handle the configuration of the workflows
- Command line interface for running the workflows
- Outline of project structure and utilities


**Dependencies**

Added requirements:

- gdal>=3.3
- h5py>=3.6
- numba>=0.54
- numpy>=1.20
- pydantic>=1.10
- pymp-pypi>=0.4.5
- ruamel_yaml>=0.15
- scipy>=1.5

Currently, Python 3.7 is supported, but 3.11 is not due numba not yet supporting Python 3.11.
