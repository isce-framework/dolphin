# Unreleased

**Added**

- Sample test data for the `dolphin` package loaded onto Zenodo.

**Dependencies**

Added testing requirements:
- pooch


# [0.1.0](https://github.com/opera-adt/dolphin/compare/v0.0.4...v0.1.0) - 2023-03-31

- First version of the `_product.py` module to output the combined NetCDF product file.
- `_pge_runconfig.py` module to handle the separate PGE-compatible configuration, which translates to-from the `Workflow` object.
- `docker/build-docker-image.sh` script to build the docker image.
- Release scripts for generating documentation, script for validating output data by @gmgunter .
- Use of a spatial correlation estimate for unwrapping purposes, rather than temporal coherence.
  - This is much more useful when the stack size is small (high temporal coherence), and `snaphu` is used for unwrapping.
- `masking.py` module for masking the interferogram/combined multiple external masks of varying 1/0 conventions.
- Ability to use existing amplitude mean/dispersion files for the PS portion of the workflow, skipping the step where we compute it using the SLC stack. Useful for small stack sizes
- Added a `create_only` option to `write_arr` to create an empty file without writing data (e.g. to check the boundary results of stitching)


**Changed**
- The YAML output/input functions are moved to a `YamlModel` class, which is a subclass of `pydantic.BaseModel`.
  - This allows us to use it in both `config.py` and `_pge_runconfig.py`.
- Refactoring of the `Workflow` layout to more easily extract the input/output files for the PGE run.

**Fixed**

- Compressed SLC outputs were getting corrupted upon writing when using strides > 1.
- Single-update interferograms where the first SLC input is a compressed SLC was broken (using the wrong size raster).
  - Now the result will simply copy over the phase-linking result, which is already referenced to the first raster.

**Dependencies**

Added requirements:

- h5netcdf>=1.1
- Avoid HDF5 version 1.12.1 until NetCDF loading issue is fixed

# [0.0.4](https://github.com/opera-adt/dolphin/compare/v0.0.3...v0.0.4) - 2023-03-17

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
- Replace `tqdm` with `rich` for progress bars.
- The `unwrap.py` module now uses isce3 to unwrap the interferogram.

- Docs are now using the mkdocs `material` theme.

**Removed**

- `utils.parse_slc_strings` in favor of always using `utils.get_dates`.
- `io.get_stack_nodata_mask`. This will be done using the nodata polygon, or not at all.


**Dependencies**

Added requirements:

- pyproj>=3.2
- rich>=12.0
- threadpoolctl>=3.0
- isce3>=0.8.0
- pyproj>=3.3
- Dropped support for Python 3.7

For docs:
- mkdocs-material
- pymdown-extensions

Removed requirements:

- tqdm


# [0.0.3](https://github.com/opera-adt/dolphin/compare/v0.0.2...v0.0.3) - 2023-01-26

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

# [0.0.2](https://github.com/opera-adt/dolphin/compare/v0.0.1...v0.0.2) - 2023-01-24

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


# [0.0.1] - 2022-12-09

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
