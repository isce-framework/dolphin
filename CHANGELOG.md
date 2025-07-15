# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased](https://github.com/isce-framework/dolphin/compare/v0.40.0...HEAD)

## [0.40.0](https://github.com/isce-framework/dolphin/compare/v0.39.0...v0.40.0) - 2025-06-

### Fixed

- Throw a better error for `dolphin` without any args

### Changed

- Remove `avg_coh` from optional phase linking outputs
- `unwrap.py`: Combine sliding window mask with similarity mask for masking / interpolation

## [0.39.0](https://github.com/isce-framework/dolphin/compare/v0.38.0...v0.39.0) - 2025-06-09

Largest visible change: Switch CLI to use `tyro` by @scottstanie in https://github.com/isce-framework/dolphin/pull/578
This involves multiple fixes and some breaking changes:

- Fixed
  - Missing arguments that were not configurable via the command line
  - `dolphin unwrap` and `dolphin timesries` has several issues resulting from the function API being out of sync with the argparse API
- `config` Changes:
  - Strides must be specified as `--sx 6 --sy 3`, not `--strides 6 3`
  - Several worker settings options like `--threads-per-worker` are not longer top-level command line options; you can specify them with `--worker-settings.threads-per-worker`

### Added

- Parallelize phase linking for single-swath case in `single.py` by @scottstanie in https://github.com/isce-framework/dolphin/pull/576
- DOC: Add phase linking theory notebook to docs by @scottstanie in https://github.com/isce-framework/dolphin/pull/597
- Add Github link to docs by @scottstanie in https://github.com/isce-framework/dolphin/pull/575
- Add script to read GAMMA `.rslc` binary SLCs by @scottstanie in https://github.com/isce-framework/dolphin/pull/569
- Add `keep_bits` to `BackgroundBlockWriter` API so writers can `round_mantissa` by @scottstanie in https://github.com/isce-framework/dolphin/pull/536

### Changed

- Make `reference` optional for velocity fitting by @scottstanie in https://github.com/isce-framework/dolphin/pull/571
- Allow `Iterator[Filename]` to all readers with `.from_file_list` by @scottstanie in https://github.com/isce-framework/dolphin/pull/579
- Fix to be `Iterable`, not `Iterator` API change for `_readers` from #579 by @scottstanie in https://github.com/isce-framework/dolphin/pull/582
- Simplify reference index logic for `ALWAYS_FIRST` in `stack.py`, cut `FIRST_PER_MINISTACK` by @scottstanie in https://github.com/isce-framework/dolphin/pull/588
- Remove single-burst special case in phase linking outputs by @scottstanie in https://github.com/isce-framework/dolphin/pull/574
- Add `fmt=file_date_fmt` to all `get_dates` call to avoid file format assumptions by @scottstanie in https://github.com/isce-framework/dolphin/pull/585

### Fixed

- Run `fix_typos.sh` on repo by @scottstanie in https://github.com/isce-framework/dolphin/pull/572
- Remove per-burst phase linking dirs that are empty by @scottstanie in https://github.com/isce-framework/dolphin/pull/595
- `stitching.py`: fix strides check for either x or y by @scottstanie in https://github.com/isce-framework/dolphin/pull/596
- Pass through `timeseries_options.reference_point` during `displacement.py` workflow by @scottstanie in https://github.com/isce-framework/dolphin/pull/602
- `unwrapping.py`: fix comparison to avoid mask warping by @scottstanie in https://github.com/isce-framework/dolphin/pull/600

### Removed

- Remove unused `_utils` helper functions by @scottstanie in https://github.com/isce-framework/dolphin/pull/581
- `OutputOptions`: Only require `bounds_epsg` if specifying `bounds`, cut `resolution` by @scottstanie in https://github.com/isce-framework/dolphin/pull/591
- Remove old `glrt_cutoffs.csv` from `MANIFEST.in` by @scottstanie in https://github.com/isce-framework/dolphin/pull/573

## [0.38.0](https://github.com/isce-framework/dolphin/compare/v0.37.0...v0.38.0) - 2025-04-15

### Added

- `demo-filtering-sizes.ipynb` for high-pass filtering cutoff demo by @scottstanie in https://github.com/isce-framework/dolphin/pull/561

### Changed

- **Breaking change**: Split corrections, remove raider pulled modules by @scottstanie in https://github.com/isce-framework/dolphin/pull/531

### Fixed

- Use `is_symlink` when checking for existing scratch files in spurt by @scottstanie in https://github.com/isce-framework/dolphin/pull/568

## [0.37.0](https://github.com/isce-framework/dolphin/compare/v0.36.2...v0.37.0) - 2025-03-27

### Changed

- `filtering.py`: Change default `sigma` calculation, add docstring explanation by @scottstanie in https://github.com/isce-framework/dolphin/pull/560

## [0.36.2](https://github.com/isce-framework/dolphin/compare/v0.36.1...v0.36.2) - 2025-03-25

### Fixed

- Update notebook walkthrough for new output structure by @scottstanie in https://github.com/isce-framework/dolphin/pull/558
- Adding subdataset as input arguments because it is different for S1 and NISAR by @mirzaees in https://github.com/isce-framework/dolphin/pull/559

## [0.36.1](https://github.com/isce-framework/dolphin/compare/v0.36.0...v0.36.1) - 2025-03-06

### Fixed

- Fix ionosphere reading and delay computation by @scottstanie in https://github.com/isce-framework/dolphin/pull/554
- Let amp_dispersion_threshold be 0 to select no PS points by @scottstanie in https://github.com/isce-framework/dolphin/pull/553
- Move `WriteArray` calls after `SetNoDataValue` calls by @scottstanie in https://github.com/isce-framework/dolphin/pull/555

### Added

- Add nisar wavelengths to `constants` by @mirzaees in https://github.com/isce-framework/dolphin/pull/551

## [0.36.0](https://github.com/isce-framework/dolphin/compare/v0.35.1...v0.36.0) - 2025-02-21

### Added

- `utils.grow_nodata_region` to remove bad borders from images

### Changed

- `timeseries.py`: Auto reference point selection
  - Picks the center of mass instead of arbitrary `argmax` result
  - Rename `condition_file` to `quality_file`
- Convert `glrt` SHP method to use JAX instead of Numba
- Remove import for `goldstein` and `filtering` from top level `__init_.py`

### Removed

- Removed `condition` parameter in `timeseries` reference point functions
- Removed `CallFunc` enum

### Fixed

- docs: Fix mathjax render, fix broken `DateOrDatetime` error
- Replace `__name__` with "dolphin" in `getLogger` to fix missing log entries from `.jsonl` log files
- Used `utils.grow_nodata_region` to remove bad borders from `shp_counts.tif`

## [0.35.1](https://github.com/isce-framework/dolphin/compare/v0.35.0...v0.35.1) - 2025-01-15

### Fixed

- `filtering.py` Fix in_bounds_pixels masking, set default to 25 km
- Set `output_reference_idx` separately from `compressed_reference_idx` during phase linking setup

## [0.35.0](https://github.com/isce-framework/dolphin/compare/v0.34.0...v0.35.0) - 2025-01-09

### Added

- Compute timeseries inversion and save to rasters
- `use_seasonal_coherence` parameter to simulate.py

### Changed

- `keepdims` options to reader classes to avoid squeezing singleton dims
- Using L1 inversion by default for timeseries
- Using most recent compressed SLC as output for `LAST_PER_MINISTACK`
- Adapted short wavelength filter to use `gdal_fillnodata` for edge effects mitigation
- Removed Numba dependency from `simulate.py`

### Fixed

- Documentation grammar on VRT size check

## [0.34.0](https://github.com/isce-framework/dolphin/compare/v0.33.0...v0.34.0) - 2024-11-26

### Added

- `--use-evd` option to `dolphin config` CLI
- `CITATION.cff` file for better zenodo parsing

### Changed

- Spurt changes
  - Logging redirect, add a log file, multiprocess during final interpolation
  - Improved speed of post-spurt 2-pi ambiguity interpolation
- Default ministack_size changed to 15 in PhaseLinkingOptions
- Using nearest sampling for stitching static layers

### Fixed

- Unit string written to `velocity.tif`
- `extra_reference_date` logic for single reference unwrapping
- Missing annotations for Python 3.9 compatibility
- Pinned `mkdocs-jupyter` to 0.25.0
- Make `--condition-file` not required for `dolphin timeseries`
- Removed `.flake8`

## [0.33.0](https://github.com/isce-framework/dolphin/compare/v0.32.0...v0.33.0) - 2024-11-07

### Added

- `fill_value` parameter to `filter_long_wavelength`
- Function to create counts of valid unwrapped outputs (`create_nonzero_conncomp_counts`)

### Changed

- Modified logging capture from spurt to be more selective

## [0.32.0](https://github.com/isce-framework/dolphin/compare/v0.31...v0.32.0) - 2024-11-05

### Fixed

- Merge documentation improvements from `joss` branch
- Add a retry to spurt for `BrokenProcessPools`

## [0.31](https://github.com/isce-framework/dolphin/compare/v0.30...v0.31) - 2024-11-03

### Changed

- Skip nodata values of interferograms when interpolating


## [0.30](https://github.com/isce-framework/dolphin/compare/v0.29.0...v0.30) - 2024-11-01

### Added
- Support for specifying output bounds as WKT
- Ability to calculate similarity using nearest-3 interferograms
- Add configuration option for single tile reoptimize functionality for snaphu-py. Turn off by default.
- Support for layover shadow mask files to mask pixels during wrapped phase
- Zero correlation threshold parameter for phase linking
- Censored least squares solving capability for missing data
- Support for reading from S3 using osgeo.gdal with `/vsis3` conversion

### Changed
- Updated dolphin timeseries CLI for new options, using L1 by default
- Increased spurt `max_tiles` default to 49 for smaller MCF problems
- Modified compressed_reference_idx with relative index fix
- Replaced CSVs with `chi2.ppf` for GLRT test
- Lowered `min_conncomp_frac` default to 0.001
- Using `output_reference_idx` as default for creating compressed SLCs with `ALWAYS_FIRST`
- Increased `buffer_pixels` when making OPERA CSLC mask
- Refactored `repack_raster` to work in blocks for lower memory usage

### Fixed
- Fixed ministack output and compressed SLC indexing
- Fixed `in_trim_slice` size for non-overlapping blocks
- Fixed units passthrough during timeseries._redo_reference
- Improved spurt subprocess handling to avoid fork issues
- Fixed block_shape passthrough to create_similarities after phase linking

## [0.29.0](https://github.com/isce-framework/dolphin/compare/v0.28.0...v0.29.0) - 2024-10-14

### Added
- Sample dolphin_config.yaml to the documentation
- Stitched phase cosine similarity raster to outputs
- Post-processing step to fill NaN gaps after running spurt
- Outside input range options to get_nearest_date_idx

### Changed
- Set EVD input to be weighted by correlation
- Using dataclasses instead of NamedTuple for displacement and stitched outputs
- Passing through phase_linking.output_reference_idx to avoid reference resets

### Fixed
- Added logic to reset extra_reference_date after network unwrapping
- Applied nan/0 mask to output of goldstein filter
- Added references/url fixes from JOSS branch

## [0.28.0](https://github.com/isce-framework/dolphin/compare/v0.27.1...v0.28.0) - 2024-09-19

### Added
- `num_parallel_tiles` and `num_parallel_ifgs` to `SpurtOptions`
- L1-norm-based inversion to `timeseries`
- Config option to avoid PS pixels during phase linking
- `block_shape` to timeseries parameters
- `wavelength` attribute to workflow to convert rasters to meters
- Larger `N` rows to computed GLRT cutoffs
- Logic for manually specifying reference dates mid-stack
- ADMM to minimize L1-norm of `Ax - b`
- Instructions for building Docker image in documentation
- Link to notebook in `simulate-demo` function
- Printing of unwrapping options in `show_versions`
- Update walkthrough notebook for docs
- Add clarifications and suggested changes to documentation

### Changed
- Remove transposed versions of functions in `covariance`
- Set `use_slc_amp` default to False during phase linking
- Use the same mask for unwrapping and conncomp regrowing steps
- Turn off correlation weighting for velocity by default
- Use less memory for L1 inversion
- Include layover shadow mask when stitching in `prepare_geometry`
- Create `.conncomp` files for `spurt` from temporal coherence
- Add `like_filename` for `spurt` connected components

### Fixed
- Fix plotting/imports for walkthrough
- Fix `displacement.run` to avoid troposphere computation with empty list
- Pass through EPSG code to `_get_mask`
- Fix `n_parallel_tiles` to be an integer, not float
- Skip `combine_mask_files` if the output already exists
- Fix typos in walkthrough configuration
- Fix `simulate-demo` function usage
- Fix mismatching docs and missing log statements
- Pass through `add_overviews`, run even if single-reference network
- Add `ignore_errors=True` to `rmtree` for scratch removal
- Setup `gaussian_filter_nan` in utils, use in `estimate_correlation_from_phase`
- Fix Apache end template
- Fix ionospheric date parsing and grouping
- Fix handling of `nodata` during timeseries conversion to meters

## [0.27.1](https://github.com/isce-framework/dolphin/compare/v0.27.0...v0.27.1) - 2024-08-12

### Changed
- Set the default `input_conventions` to be same as `output_conventions`

## [0.27.0](https://github.com/isce-framework/dolphin/compare/v0.26.0...v0.27.0) - 2024-08-12

### Added
- Ability to mask a subset of input SLCs

### Changed
- Temporarily removed resampling during stitching final crop


## [0.26.0](https://github.com/isce-framework/dolphin/compare/v0.25.0...v0.26.0) - 2024-08-02

### Added
- `map` option for `DummyProcessPoolExecutor`
- PR template
- `dolphin filter` CLI to run a long wavelength filter on a set of rasters

### Changed
- Changed datetime format for reading tropospheric corrections
- Copy over unwrapped to `timeseries/` even with 1 file
- Removed `atmosphere` imports from the main path

## [0.25.0](https://github.com/isce-framework/dolphin/compare/v0.24.0...v0.25.0) - 2024-08-01

### Changed
- Moved reference point selection back to always be run if not specified
- Reference correction outputs to a point in space and convert to meters

### Fixed
- Fixed `timeseries` to mkdir right away

## [0.24.0](https://github.com/isce-framework/dolphin/compare/v0.23.0...v0.24.0) - 2024-08-01

### Added
- Skip correlation estimate for files that exist
- Save `ReferencePoint` into file in `timeseries/` folder

### Changed
- Clip `nlooks` to a max of 20 for `whirlwind`
- Refactor `filter_long_wavelength` to avoid boundary issues for images with lots of no data region
- Skip zero raster output by timeseries inversions

### Fixed
- Fixed reference point selection labels


## [0.23.0](https://github.com/isce-framework/dolphin/compare/v0.22.1...v0.23.0) - 2024-07-29

### Added
- [`whirlwind`](https://github.com/isce-framework/whirlwind) unwrapper
- Option to redirect other loggers to same JSON file
- Interleave GTiffs by band, not pixel

### Changed
- Updated `show_versions` to read `importlib.metadata`, use `keep_bits` instead of `lerc`
- Made network formation `staticmethod`s public
- Shrunk `nlooks` for whirlwind
- Improved code for transferring unwrapped phase ambiguities
- Moved mask loading function into `masking.py`

### Fixed
- Fixed Spurt tests
- Updated `spurt` imports for exported interface
- Set the `final_arr` nodata to be same as original ifg mask
- Fixed filtering ambiguity transfer for interp/goldstein
- Fixed confusing name for scratchdirs

## [0.22.1](https://github.com/isce-framework/dolphin/compare/v0.22.0...v0.22.1) - 2024-07-17

### Fixed

- Fixed v0.22.0 imports

## [0.22.0](https://github.com/isce-framework/dolphin/compare/v0.21.0...v0.22.0) - 2024-07-16

### Added
- Output velocities in units per year in `timeseries`

### Changed
- Minor changes in `spurt` config to match its naming conventions
- Vendored `mdx_bib` to avoid direct URL in `docs/requirements.txt`

## [0.21.0](https://github.com/isce-framework/dolphin/compare/v0.20.0...v0.21.0) 2024-07-16

### Added
- Continuous deployment Github action workflow to publish to pypi
- Conversion from NetCDF tropospheric correction files

### Changed
- Round mantissa bits instead of truncate, call the argument `keep_bits` for clarity

## [0.20.0](https://github.com/isce-framework/dolphin/compare/v0.19.0...v0.20.0) - 2027-07-08

### Added
- Functions in `ps` to combine amplitude dispersions from older rasters
- `truncate_mantissa` option to `repack` functions for simple compression
- `band` option to `write_block` and background writers
- Option to run `merge_dates` and `estimate_interferometric_correlations` with `thread_map` for parallel processing
- Baseline lag option for "STBAS" phase linking inversion

### Changed
- Logging now uses `dictConfig` and always logs JSON to a file for the Displacement workflow
- Set modulus of compressed SLCs to be real SLC magnitude means
- Updated Docker requirements and specfile
- Delete intermediate unwrapping `scratchdir`s by default

### Fixed
- `use_max_ps` would occasionally fail with certain stride/window configurations
- Unwrapped phase files did not always contain the right geographic metadata
- Filenames in the `timeseries/` folder were wrong
- Set upsampled boundary to `nan` in `compress`
- Unwrapped file output path

## [0.19.0](https://github.com/isce-framework/dolphin/compare/v0.18.0...v0.19.0) - 2024-06-21

### Added
- `filtering` module for filtering out long wavelength signals from the unwrapped phase
- `baseline` module for computing the perpendicular baseline. Initial version has logic for OPERA CSLCs, uses `isce3`
- Interface only for 3D unwrapping
- Faster correlated noise simulation, along with 3d stack simulation with synthetic deformation
- Added ability to read rasters on S3 using `VRTStack` object
- Eigenvalue solver speedups of 3-9x
- Initial version of 3D unwrapping using `spurt`

### Removed
- the KL-divergence SHP estimator has been removed. GLRT is recommended instead.

### Fixed
- `reproject_bounds` uses the `rasterio` version, which densifies points along edges for a more accurate bounding box
- The output SHP rasters now output 0 if there was no valid input data
- Logic for filling PS pixels, with and without setting the amplitudes to be the original SLC amplitudes
- `ReferencePointError` during the displacement workflow now will fall back look only at the `condition_file` (i.e. choose the point with highest temporal coherence by default)

### Changed
- The configuration options for unwrapping have been refactored. Options unique to each unwrapper are grouped into subclasses.
  - Note that older `dolphin_config.yaml` files will error after this refactor.
- Unweighted time series inversion will make one batch call, providing a large speedup over the `vmap` version for weighted least squares

## [0.18.0](https://github.com/isce-framework/dolphin/compare/v0.17.0...v0.18.0) - 2024-05-07

### Added
- `dolphin timeseries` command line tool for inverting unwrapped interferogram network and estimating velocity

### Fixed
- Parse the file names correctly to find compressed SLCs and read dates based on production file naming convention


## [0.17.0](https://github.com/isce-framework/dolphin/compare/v0.16.3...v0.17.0) - 2024-04-10
### Added
- Added Goldstein filtering for unwrapping
- Added Interpolation for unwrapping
- Added the regrow connected components for the modified phase
- Added option to toggle off inversion
- Added similarity module

### Fixed
- 3D readers would squeeze out a dimension for length one inputs (i.e. they would give an array with `.ndim=2`)
- `max_bandwidth` config can now be 1 to specify only nearest neighbor interferograms.
- Use the 'compressed' key term to find compressed slcs and regular slcs instead of number of dates in ionosphere
- Consider the compressed SLCs have different naming convention with capital letters
- Enforce consistency between jax and jaxlib
- Disable corrections part of pytest, add one for timeseries

## [0.16.0](https://github.com/isce-framework/dolphin/compare/v0.15.3...v0.16.0) - 2024-03-03

### Added
- Added `dolphin.timeseries` module with basic functionality:
  - Invert a stack of unwrapped interferograms to a timeseries (using correlation weighting optionally)
  - Estimate a (weighted) linear velocity from a timeseries
- Added inversion and velocity estimation as options to `DisplacementWorkflow`
- Create `DatasetStackWriter` protocol, with `BackgroundStackWriter` implementation

### Changed
- Rename `GdalWriter` to `BackgroundBlockWriter`
- Displacement workflow now also creates/returns a stitched, multi-looked version of the amplitude dispersion

### Fixed
- `BackgroundRasterWriter`  was not creating the files necessary before writing
- Allow user to specify more than one type of interferogram in `Network` configuration

# [0.15.3](https://github.com/isce-framework/dolphin/compare/v0.15.2...0.15.3) - 2024-02-27

### Changed
- Return the output paths created by the ionosphere/troposphere modules to make it easier to use afterward

# [0.15.2](https://github.com/isce-framework/dolphin/compare/v0.15.1...0.15.2) - 2024-02-27

### Fixed

- Fixes to ionosphere/troposphere correction in for `DisplacementWorkflow` and `PsWorkflow`
- Correct the nodata value passed through to snaphu-py

# [0.15.1](https://github.com/isce-framework/dolphin/compare/v0.15.0...0.15.1) - 2024-02-26

### Fixed

- PHASS now uses the Tophu wrapper to avoid isce3 inconsistencies between argument order

# [0.15.0](https://github.com/isce-framework/dolphin/compare/v0.14.1...0.15.0) - 2024-02-16

### Changed

- Combine the nodata region with the `mask_file` to pass through to unwrappers
- Update regions which are nodata in interferograms to be nodata in unwrapped phase
- Use `uint16` data type for connected component labels

### Fixed

- Intersection of nodata regions for SLC stack are now all set to `nan` during phase linking, avoiding 0 gaps between bursts

# [0.14.1](https://github.com/isce-framework/dolphin/compare/v0.14.0...0.14.1) - 2024-02-15

### Fixed

- Changed snaphu-py tile defaults to avoid max secondary arcs error in #233
- Fixed `linalg.norm`` to be pixelwise in `process_coherence_matrices` in #234


# [0.14.0](https://github.com/isce-framework/dolphin/compare/v0.13.0...0.14.0) - 2024-02-13

### Fixed
- Temporal coherence and eigenvalue rasters were switched in their naming
- Output a better `estimator` raster to see where we switched to EVD
- Cap the max number of threads to the CPU count to avoid `numba` config errors

### Changed
- refactor temporal coherence calculation to use `vmap`
  - Allows us to start making a weighted temporal coherence metric
- Turn off default `beta=0.01` regularization now that CPL is in place
- Removed `InterferogramNetworkType` from configuration.
  - You can add multiple types of network parameters and it includes all of them. Adding the name only decreased flexibility.

# [0.13.0](https://github.com/isce-framework/dolphin/compare/v0.12.0...v0.13.0) - 2024-02-09

### Added

- `_overviews` module, and workflow configuration to create overviews of the output stitched rasters
- Configuration to use the [snaphu-py](https://github.com/isce-framework/snaphu-py) wrapper, and drop using `isce3.unwrap.snaphu`

### Fixed

- Apply bounds even if only one image is passed to `stitching` (#210)
- Allow `take_looks` to work with `MaskedArrays` without converting to `np.ndarray`

### Dependencies

- Move back to `tqdm` instead of using `rich` for progress bars.

## [0.12.0](https://github.com/isce-framework/dolphin/compare/v0.11.0...v0.12.0) - 2024-02-01

### Added
- Added `DatasetWriter` protocol
- Added `RasterWriter` and `BackgroundRasterWriter` implementations of this protocol
- Refactored phase linking
  - Covariance and EVD/MLE use `jax`
  - This combines the implementation of CPU/GPU, and removes the need for using `pymp`
- Added `utils.disable_gpu` to stop the use ofr a GPU even if it's available

### Changed
- Internal module organization, including grouping IO modules into `dolphin.io` subpackage
- Renamed `io.Writer` to `io.GdalWriter` to distinguish from `RasterWriter`
- Removed the `n_workers` option from the configuration.
  - There is no more need to have two levels of parallelism (`threads_per_worker` and `n_workers`)
  - The name `threads_per_worker` is kept for consistency; it is still an accurate name for the multi-burst processing case.


### Added
- `jax`>=0.4.19
- `numpy`=>1.23 (bump to minimum version that they're still supporting)
- `scipy`>=1.9 (same reason)
- `numba`>=0.54

### Removed
- `pymp`
- `cupy` from optional GPU usage

## [0.11.0](https://github.com/isce-framework/dolphin/compare/v0.10.0...v0.11.0) - 2024-01-24

### Added
- Added ionospheric correction in `dolphin.atmosphere.ionosphere`
  - Included in `DisplacementWorkflow` if TEC files are provided

## [0.10.0](https://github.com/isce-framework/dolphin/compare/v0.9.0...v0.10.0) - 2024-01-22

### Added
- Create `dolphin.unwrap` subpackage to split out unwrapping calls, and post-processing modules.

### Removed
- the `_dates` module has been removed in favor of using `opera_utils._dates`

### Fixed
- `stitching.merge_images` will now give consistent sizes for provided bounds when `strides` is given

## [0.9.0](https://github.com/isce-framework/dolphin/compare/v0.8.0...v0.9.0) - 2024-01-10

### Added
- `DatasetReader` and `StackReader` protocols for reading in data from different sources
  - `DatasetReader` is for reading in a single dataset, like one raster image.
  - `StackReader` is for reading in a stack of datasets, like a stack of SLCs.
  - Implementations of these have been done for flat binary files (`BinaryReader`), HDF5 files (`HDF5Reader`), and GDAL rasters (`RasterReader`).

### Changed
- The `VRTStack` no longer has an `.iter_blocks` method
  - This has been replaced with creating an `EagerLoader` directly and passing it to the `reader` argument

### Dependencies
- Added `rasterio>=1.3`

## [0.8.0](https://github.com/isce-framework/dolphin/compare/v0.7.0...v0.8.0) - 2024-01-05

### Added
- Ability to unwrap interferograms with the [`snaphu-py`](https://github.com/isce-framework/snaphu-py) (not a required dependency)
- Added ability to make annual ifgs in `Network`
- Start of tropospheric correction support in `dolphin.atmosphere` using PyAPS and Raider packages
- Expose the unwrap skipping with `dolphin config --no-unwrap`

### Changed
- The output directory for interferograms is now just "interferograms/" instead of "interferograms/stiched"
  - Even when stitching, the burst-wise interferograms would be in the named phase-linking subfolders.
- Split apart the `dolphin.workflows.stitch_and_unwrap` module into `stitching_bursts` and `unwrapping`
- Switched output filename from `tcorr` to `temporal_coherence` for the temporal coherence of phase linking.
  - Also added the date span to the `temporal_coherence` output name
- The default extension for conncomps is now `.tif`. Use geotiffs instead of ENVI format for connected components.
- Using ruff instead of pydocstyle due to archived repo

## [0.7.0](https://github.com/isce-framework/dolphin/compare/v0.6.1...v0.7.0) - 2023-11-29

### Added
- `MiniStackPlanner` and `MiniStackInfo` class which does the planning for how a large stack of SLCs will be processed in batches.
  - Previously this was done at run time in `sequential.py`. We want to separate that out to view the plan in advance/allow us to dispatch the work to multiple machines.
- `CompressedSlcInfo` class added to track the attributes of a compressed SLC file created during the workflow.
  - This has the `reference_date` as an attribute, which allows us to know what the base phase is even without starting from
    the first SLC in the stack (i.e. if we have limited the number of compressed SLCs)
- Added better/more complete metadata to the compressed SLC Geotiff tags, including the phase reference date
  - Before we were relying on the filename convention, which was not enough information
- config: `phase_linking.max_compressed_slcs` to cap the number of compressed SLCs added during large-stack sequential workflows
- `interferogram`: Add ability to specify manual dates for a `Network`/`VRTInterferogram`, which lets us re-interfere the phase-linking results

### Changed
- Date functions have been moved from `dolphin.utils` to `dolphin._dates`. They are accessible at `dolphin.get_dates`, etc
- `get_dates` now uses `datetime.datetime` instead of `datetime.date`.
  - This is to allow for more flexibility in the date parsing, and to allow for the use of `datetime.date` or `datetime.datetime` in the output filenames.
- `VRTStack` has been moved to `_readers.py`. The minstack planning functions have been removed to focus the class on just reading input GDAL rasters.

### Fixed
- When starting with Compressed SLCs in the list of input SLCs, the workflows will now recognize them, find the correct reference date, and form all the correct interferograms

### Removed
- Extra subsetting functions from `VRTStack` have been removed, as they are not used in the workflow and the reimplmenent simple GDAL calls.
- `CPURecorder` and `GPURecorder` have been removed to simplify code. May be moved to separate repo.

## [0.6.1](https://github.com/isce-framework/dolphin/compare/v0.6.0...v0.6.1) - 2023-11-13

### Removed
- `dolphin.opera_utils` now lives in the separate package

### Dependencies
- Added `opera_utils`

## [0.6.0](https://github.com/isce-framework/dolphin/compare/v0.5.1...v0.6.0) - 2023-11-07

### Added
- `opera_utils.get_missing_data_options` to parse the full list of SLCs and return possible subsets which have the same dates used for all Burst IDs
- `PsWorkflow` class for running just the PS estimation workflow
- `asv` benchmark setup to measure runtime across versions
- `@atomic_output` decorator for long running write processes, to avoid partially-written output files

### Changed
- removed `minimum_images` as an argument from `opera_utils.group_by_burst`. Checking for too-few images now must be done by the caller
- `opera_utils.group_by_burst` now matches the official product name more robustly, but still returns the lowered version of the burst ID.
- The `s1_disp` workflow has been renamed to `displacement`, since it is not specific to Sentinel-1.
- The configuration was refactored to enable smaller workflow
  - The `Workflow` config class has been renamed to `DisplacementWorkflow`.
  - A `PsWorkflow` config class has been added for the PS estimation workflow.
  - A `WorkflowBase` encompasses some of the common configuration options.

### Maintenance
- `ruff` has replaced `isort`/`black`/`flake8` in the pre-commit checks

## [0.5.1](https://github.com/isce-framework/dolphin/compare/v0.5.0...v0.5.1) - 2023-10-10

### Added
- `stitch_and_unwrap.run` returns the stitch PS mask

## [0.5.0](https://github.com/isce-framework/dolphin/compare/v0.4.3...v0.5.0) - 2023-10-09

### Added
- `CPURecorder` class for fine grained benchmarking of the CPU/memory usage for

### Changed
- Docker `specfile` now builds with tophu

## [0.4.3](https://github.com/isce-framework/dolphin/compare/v0.4.2...v0.4.3) - 2023-10-06

### Added
- Ability to unwrap using isce3's `PHASS`
- `CorrectionOptions` model for specifying the correction options in the `Workflow` config
  - Currently a placeholder for the files which will be used for tropospheric/ionospheric corrections
- Ability to keep relative files in the `Workflow` config
  - This is useful for keeping the relative paths to the SLCs in the config, and then running the workflow from a different directory

### Changed

- Instead of specifying the unwrapping algorithm in `dolphin unwrap` as `--use-icu`, the option is not `--unwrap-method`
  - This let's us add `--unwrap-method "phass"`, but also future unwrap methods without a `--use-<name>` for every one
- Use `spawn` instead of `fork` for parallel burst multiprocessing
  - This leads to the error `Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.`
    in certain situations, and does not happen with `spawn`. See https://pythonspeed.com/articles/python-multiprocessing/ for more details.


# [0.4.2](https://github.com/isce-framework/dolphin/compare/v0.4.1...v0.4.2) - 2023-10-03

### Added
- `use_evd` option to force the use of eigenvalue decomposition instead of the EMI phase linking algorithm
- Walkthrough tutorial notebook

### Changed

- Moved all `OPERA_` variables to a new module `dolphin.opera_utils`.
  - Other OPERA-specific quirks have been moved to the separate `disp-s1` repo,
     but the functions remaining are the ones that seem most broadly useful to `sweets`
     and other users working with burst SLCs.
  - Changed the burst regex to be able to match COMPASS and the official product name
- Removed `WorkflowName` for separating `stack` vs `single`
  - The name didn't really provide benefit, as the real differences cam from other configuration options
- Internals for which functions are called in `sequential.py`
- Docker image now has `tophu` installed

## [0.4.1](https://github.com/isce-framework/dolphin/compare/v0.4.0...v0.4.1) - 2023-09-08

### Dependencies
- Added back isce3

## [0.4.0](https://github.com/isce-framework/dolphin/compare/v0.3.0...v0.4.0) - 2023-09-07


### Changed

- Split apart OPERA-specific needs from more general library/workflow functionality
- Removed the final NetCDF product creation
  - Many rasters in the `scratch/` folder are of general interest after running the workflow
  - Changed folder structure so that there's not longer a top-level `scratch/` and `output/` by default
- Changed the required dependencies so the `isce3` unwrapper is optional, as people may wish to implement their own custom parallel unwrapping

### Dependencies

Dropped:
- h5netcdf
- pillow

Now optional:
- isce3 (for unwrapping)

## [0.3.0](https://github.com/isce-framework/dolphin/compare/v0.2.0...v0.3.0) - 2023-08-23

### Added

- Save a multilooked version of the PS mask for output inspection

### Changed

- Pydantic models were upgraded to V2
- Refactored the blockwise IO into `_blocks.py`.
  - The iteration now happens over the output grid for easier dilating/padding when using `strides`
  - New classes with `BlockIndices` and `BlockManager` for easier management of the different slices

### Dependencies

- pydantic >= 2.1

## [0.2.0](https://github.com/isce-framework/dolphin/compare/v0.1.0...v0.2.0) - 2023-07-25

### Added

- For OPERA CSLC inputs, we now read the nodata polygon and skip loading regions of the SLC stack which are all nodata.
  - This led to a reduction of 30-50% in wrapped phase estimation runtime for each burst stack.
- Sample test data for the `dolphin` package loaded onto Zenodo.
- Adds 3 methods of computing a variable statistically homogeneous pixel (SHP) window when estimating the covariance matrix:
  - Kolmogorov-Smirnov test (KS-test)
  - Generalized likelihood ratio test (GLRT)
  - Kullback-Leibler divergence/distance test (KLD)
  - `"rect"` is also an option for skipping any statistical test and using the full rectangular multilook window
- Also included a script to view the window in an interactive matplotlib figure (matplotlib must be installed separately)
- Added a simple method to check for adjacent-pixel unwrapping errors in `unwrap.compute_phase_diffs`
- Adds a method `utils.get_cpu_count` which returns either `os.cpu_count`, or (if running in a Docker container) the number of CPUs allocated by Docker
- If processing stacks from separate bursts, added option `n_parallel_bursts` to `Workflow` to run in parallel processes.
- Created a script to test the incremental/near-real-time version of phase linking
- Added a new CLI command `dolphin unwrap` to unwrap a single interferogram/a directory of interferograms in parallel.
- Added ability to specify a glob pattern for input CSLC files in the YAML config
- Saves a multilooked PS mask for the default workflow output
- Allows the user to specify a desired bounds for the final stitched result

**Changes**

- Default OPERA dataset is now within `/data`, reflecting the new COMPASS product spec since CalVal
- Passing an existing file to `VRTStack` will no longer error unless `fail_on_overwrite=True`. The default just prints out the overwrite is happening. This prevents multiple runs in the same folder from errorings just for creating a reference to the SLC files.
- The environment variable `NUMBA_NUM_THREADS` is set using the passed in config to prevent numba from using all CPUs during `prange` calls
- The `sequential.py` module uses a different implementation of the sequential estimator to avoid the need for a datum adjustment.
- The scratch directory holding unwrapped interferograms is named `unwrapped` instead of `unwrap`
- Stitching files now can accept downsampled versions and product the correct geo metadata

### Fixed

- Calculating the nodata mask using the correct input geotransform
- Trims the overlapped region of the phase linking step when iterating in blocks

### Dependencies

- shapely >= 1.8
- Numba now supports Python 3.11, so we can drop the Python<3.11 version restriction.

Added testing requirements:
- pooch
- pillow>=7.0


## [0.1.0](https://github.com/isce-framework/dolphin/compare/v0.0.4...v0.1.0) - 2023-03-31

- First version of the `_product.py` module to output the combined NetCDF product file.
- `_pge_runconfig.py` module to handle the separate PGE-compatible configuration, which translates to-from the `Workflow` object.
- `docker/build-docker-image.sh` script to build the docker image.
- Release scripts for generating documentation, script for validating output data by @gmgunter .
- Use of a spatial correlation estimate for unwrapping purposes, rather than temporal coherence.
  - This is much more useful when the stack size is small (high temporal coherence), and `snaphu` is used for unwrapping.
- `masking.py` module for masking the interferogram/combined multiple external masks of varying 1/0 conventions.
- Ability to use existing amplitude mean/dispersion files for the PS portion of the workflow, skipping the step where we compute it using the SLC stack. Useful for small stack sizes
- Added a `create_only` option to `write_arr` to create an empty file without writing data (e.g. to check the boundary results of stitching)


### Changed
- The YAML output/input functions are moved to a `YamlModel` class, which is a subclass of `pydantic.BaseModel`.
  - This allows us to use it in both `config.py` and `_pge_runconfig.py`.
- Refactoring of the `Workflow` layout to more easily extract the input/output files for the PGE run.

### Fixed

- Compressed SLC outputs were getting corrupted upon writing when using strides > 1.
- Single-update interferograms where the first SLC input is a compressed SLC was broken (using the wrong size raster).
  - Now the result will simply copy over the phase-linking result, which is already referenced to the first raster.

### Dependencies

Added requirements:

- h5netcdf>=1.1
- Avoid HDF5 version 1.12.1 until NetCDF loading issue is fixed

## [0.0.4](https://github.com/isce-framework/dolphin/compare/v0.0.3...v0.0.4) - 2023-03-17

### Added

- Created first version of the single-update workflow, usable with `dolphin config --single`
- `_background.py` module as the abstract classes for background input/output with `EagerLoader` and `Writer`.
- `from_vrt_file` for the `VRTInterferogram` class.
- Arbitrary interferogram index selection in `Network` class.
- Parallel CPU eigenvector finding using `scipy.linalg.eigh`.
- PS selection for strided outputs using the average PS phase within a window (that contains multiple PS).
- Comments in the YAML file output by the `dolphin config` command.


### Changed

- The main workflow has been renamed to `s1_disp.py` to better reflect the workflow, since it can handle both single and stack workflows.
    - The `sequential.py` and `single.py` are where these differences are handled.
- More uniform naming in `io.get_raster_<X>` functions.
- The SLC compression is now done in `_compress.py` to declutter the `mle.py` module.
- Replace `tqdm` with `rich` for progress bars.
- The `unwrap.py` module now uses isce3 to unwrap the interferogram.

- Docs are now using the mkdocs `material` theme.

### Removed

- `utils.parse_slc_strings` in favor of always using `utils.get_dates`.
- `io.get_stack_nodata_mask`. This will be done using the nodata polygon, or not at all.


### Dependencies

Added requirements:

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


## [0.0.3](https://github.com/isce-framework/dolphin/compare/v0.0.2...v0.0.3) - 2023-01-26

### Added

- Ability for `VRTStack` to handle HDF5 files with subdatasets.
    - The OPERA specific HDF5 files are now supported without extra configuration.
- First stitching of interferograms in `stitching.py`.
    - Users can pass multiple SLC burst (like COMPASS bursts) per date, and the workflow will process per stack then stitch per date.
- More features for `load_gdal` to load in blocks.

### Changed

- A small amount of regularization on the coherence matrix is done before inversion during phase linking to avoid singular matrices.
- Renamed module to `_log.py`
- `workflows/wrapped_phase.py` absorbed much logic formerly in `s1_disp_stack.py`.

## [0.0.2](https://github.com/isce-framework/dolphin/compare/v0.0.1...v0.0.2) - 2023-01-24

### Added

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


## [0.0.1] - 2022-12-09

### Added

- Created the `config` module to handle the configuration of the workflows
- Command line interface for running the workflows
- Outline of project structure and utilities


### Dependencies

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
