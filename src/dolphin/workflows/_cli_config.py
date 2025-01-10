from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

from dolphin._types import TropoModel, TropoType
from dolphin.utils import get_cpu_count

from .config import (
    DisplacementWorkflow,
    ShpMethod,
    UnwrapMethod,
)


def create_config(
    *,
    outfile: Union[str, Path],
    print_empty: bool = False,
    slc_files: Optional[list[str]] = None,
    subdataset: Optional[str] = None,
    keep_paths_relative: bool = False,
    work_directory: Optional[Path] = Path(),
    mask_file: Optional[str] = None,
    zero_where_masked: bool = False,
    ministack_size: Optional[int] = 15,
    use_evd: bool = False,
    half_window_size: tuple[int, int] = (11, 5),
    shp_method: ShpMethod = ShpMethod.GLRT,
    baseline_lag: Optional[int] = None,
    amp_dispersion_threshold: float = 0.25,
    strides: tuple[int, int],
    output_bounds: tuple[int, int, int, int],
    block_shape: tuple[int, int] = (512, 512),
    threads_per_worker: int = 4,
    n_parallel_bursts: int = 1,
    enable_gpu: bool = False,
    ntiles: tuple[int, int] = (1, 1),
    downsample_factor: tuple[int, int] = (1, 1),
    no_unwrap: bool = False,
    no_inversion: bool = False,
    n_parallel_unwrap: int = 1,
    run_goldstein: bool = False,
    run_interpolation: bool = False,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    troposphere_files: Optional[list[str]] = None,
    tropo_date_fmt: str = "%Y%m%d",
    tropo_package: str = "pyaps",
    tropo_model: TropoModel = TropoModel.ERA5,
    tropo_delay_type: TropoType = TropoType.COMB,
    ionosphere_files: Optional[list[str]] = None,
    geometry_files: Optional[list[str]] = None,
    dem_file: Optional[str] = None,
    single_update: bool = False,
    log_file: Optional[Path] = None,
    amplitude_mean_files: Optional[list[str]] = None,
    amplitude_dispersion_files: Optional[list[str]] = None,
):
    """Create a config for a displacement workflow."""
    output = sys.stdout if outfile == "-" else outfile
    if print_empty:
        DisplacementWorkflow.print_yaml_schema(output)
        return

    if amplitude_dispersion_files is None:
        amplitude_dispersion_files = []
    if amplitude_mean_files is None:
        amplitude_mean_files = []
    if geometry_files is None:
        geometry_files = []
    if ionosphere_files is None:
        ionosphere_files = []
    if troposphere_files is None:
        troposphere_files = []
    if single_update:
        # create only one interferogram from the first and last SLC images
        interferogram_network = {"indexes": [(0, -1)]}
        # Override the ministack size so that only one phase linking is run
        ministack_size = 1000
    else:
        interferogram_network = {}  # Use default

    cfg = DisplacementWorkflow(
        cslc_file_list=slc_files,
        work_directory=work_directory,
        keep_paths_relative=keep_paths_relative,
        mask_file=mask_file,
        input_options={
            "subdataset": subdataset,
        },
        interferogram_network=interferogram_network,
        output_options={
            "strides": {"x": strides[0], "y": strides[1]},
            "bounds": output_bounds,
        },
        phase_linking={
            "ministack_size": ministack_size,
            "half_window": {"x": half_window_size[0], "y": half_window_size[1]},
            "shp_method": shp_method,
            "use_evd": use_evd,
            "baseline_lag": baseline_lag,
        },
        ps_options={
            "amp_dispersion_threshold": amp_dispersion_threshold,
        },
        unwrap_options={
            "unwrap_method": unwrap_method,
            "run_unwrap": not no_unwrap,
            "zero_where_masked": zero_where_masked,
            "run_goldstein": run_goldstein,
            "run_interpolation": run_interpolation,
            "n_parallel_jobs": n_parallel_unwrap,
            "snaphu_options": {
                "ntiles": ntiles,
            },
            "tophu_options": {
                "ntiles": ntiles,
                "downsample_factor": downsample_factor,
            },
        },
        timeseries_options={
            "run_inversion": not no_inversion,
        },
        correction_options={
            "troposphere_files": troposphere_files,
            "tropo_date_fmt": tropo_date_fmt,
            "tropo_package": tropo_package,
            "tropo_model": tropo_model,
            "tropo_delay_type": tropo_delay_type,
            "ionosphere_files": ionosphere_files,
            "geometry_files": geometry_files,
            "dem_file": dem_file,
        },
        worker_settings={
            "block_shape": block_shape,
            "n_parallel_bursts": n_parallel_bursts,
            "threads_per_worker": threads_per_worker,
            "gpu_enabled": enable_gpu,
        },
        log_file=log_file,
        amplitude_mean_files=amplitude_mean_files,
        amplitude_dispersion_files=amplitude_dispersion_files,
    )

    print(f"Saving configuration to {output!s}", file=sys.stderr)
    cfg.to_yaml(output)


def get_parser(subparser=None, subcommand_name="run"):
    """Set up the command line interface."""
    metadata = {
        "description": "Create a configuration file for a displacement workflow.",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
        # https://docs.python.org/3/library/argparse.html#fromfile-prefix-chars
        "fromfile_prefix_chars": "@",
    }
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)
    else:
        parser = argparse.ArgumentParser(**metadata)

    # parser._action_groups.pop()
    parser.add_argument(
        "--print-empty",
        action="store_true",
        help="Flag to print a YAML file with only default filled to `outfile`.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="dolphin_config.yaml",
        help="Name of YAML configuration file to save to. Use '-' to write to stdout.",
    )
    # Get Inputs from the command line
    inputs = parser.add_argument_group("Input options")
    inputs.add_argument(
        "--slc-files",
        nargs=argparse.ZERO_OR_MORE,
        help="List the paths of all SLC files to include.",
    )
    inputs.add_argument(
        "--keep-paths-relative",
        action="store_true",
        help=(
            "Don't resolve file paths that are given as relative (useful for running"
            " inside docker). "
        ),
    )
    # Get the subdataset of the SLCs to use, if passing HDF5/NetCDF files
    inputs.add_argument(
        "-sds",
        "--subdataset",
        help="Subdataset to use from HDF5/NetCDF files.",
    )
    inputs.add_argument(
        "--amplitude-mean-files",
        nargs=argparse.ZERO_OR_MORE,
        help="Optional: List the paths of existing amplitude mean files.",
        default=[],
    )
    inputs.add_argument(
        "--amplitude-dispersion-files",
        nargs=argparse.ZERO_OR_MORE,
        help="Optional: List the paths of existing amplitude dispersion files.",
        default=[],
    )
    inputs.add_argument(
        "--mask-file",
        help=(
            "Path to Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data."
        ),
    )

    # Phase linking options
    pl_group = parser.add_argument_group("Phase Linking options")
    pl_group.add_argument(
        "-ms",
        "--ministack-size",
        default=15,
        help="Strides/decimation factor (x, y) (in pixels) to use when determining",
    )
    pl_group.add_argument(
        "--use-evd",
        action="store_true",
        help="Use EVD for phase linking instead of EMI/CPL",
    )
    # Half window size for the phase linking algorithm
    pl_group.add_argument(
        "-hw",
        "--half-window-size",
        type=int,
        nargs=2,
        default=(11, 5),
        metavar=("X", "Y"),
        help="Half window size for the phase linking algorithm",
    )
    pl_group.add_argument(
        "--shp-method",
        type=ShpMethod,
        choices=[s.value for s in ShpMethod],
        default=ShpMethod.GLRT,
        help="Method used to calculate the SHP.",
    )
    pl_group.add_argument(
        "--baseline-lag",
        type=Optional[int],
        default=None,
        help="Half window size for the phase linking algorithm",
    )

    # PS options
    ps_group = parser.add_argument_group("PS options")
    ps_group.add_argument(
        "--amp-dispersion-threshold",
        type=float,
        default=0.25,
        help="Threshold for the amplitude dispersion.",
    )

    # Unwrap options
    unwrap_group = parser.add_argument_group("Unwrap options")
    unwrap_group.add_argument(
        "--no-unwrap", action="store_true", help="Skip running the unwrapping step."
    )
    unwrap_group.add_argument(
        "--unwrap-method",
        choices=[m.value for m in UnwrapMethod],
        default=UnwrapMethod.SNAPHU,
        help="Choice of unwrapping algorithm to use.",
    )
    unwrap_group.add_argument(
        "-t",
        "--ntiles",
        nargs=2,
        type=int,
        metavar=("row_tiles", "col_tiles"),
        default=(1, 1),
        help=(
            "(For multiscale unwrapping) Number of tiles to split interferograms into."
        ),
    )
    unwrap_group.add_argument(
        "--downsample-factor",
        nargs=2,
        type=int,
        metavar=("row_downsample", "col_downsample"),
        default=(1, 1),
        help=(
            "(For multiscale unwrapping) Factor to multilook the coarse unwrapped"
            " version."
        ),
    )
    unwrap_group.add_argument(
        "--n-parallel-unwrap",
        type=int,
        default=1,
        help="Number of interferograms to unwrap in parallel.",
    )
    unwrap_group.add_argument(
        "--run-goldstein",
        action="store_true",
        help="Run Goldstein filter before unwrapping.",
    )
    unwrap_group.add_argument(
        "--run-interpolation",
        action="store_true",
        help="Run interpolation before unwrapping.",
    )

    # Correction options
    correction_group = parser.add_argument_group("Correction options")
    correction_group.add_argument(
        "--troposphere-files",
        nargs=argparse.ZERO_OR_MORE,
        help="List the paths of all troposphere files to include.",
    )
    correction_group.add_argument(
        "--tropo-date-fmt",
        default="%Y%m%d",
        help="Format of dates contained in weather-model filenames.",
    )
    correction_group.add_argument(
        "--tropo-package",
        default="pyaps",
        help="Package to use for tropospheric correction. Choices are: pyaps, raider.",
    )
    correction_group.add_argument(
        "--tropo-model",
        default=TropoModel.ERA5.value,
        choices=[t.value for t in TropoModel],
        type=TropoModel,
        help="source of the atmospheric model.",
    )
    correction_group.add_argument(
        "--tropo-delay_type",
        default=TropoType.COMB.value,
        type=TropoType,
        help="Tropospheric delay type: comb contains both wet and dry delays",
        choices=[t.value for t in TropoType],
    )
    correction_group.add_argument(
        "--ionosphere-files",
        nargs=argparse.ZERO_OR_MORE,
        help="List the paths of all ionosphere files to include.",
    )
    correction_group.add_argument(
        "--geometry-files",
        nargs=argparse.ZERO_OR_MORE,
        help="List the paths of all geometry files to include.",
    )
    correction_group.add_argument(
        "--dem-file",
        help="Path to DEM file to use for topographic correction.",
    )

    # Get Outputs from the command line
    out_group = parser.add_argument_group("Output options")
    out_group.add_argument(
        "--single-update",
        action="store_true",
        help="Create only one interferogram from the first and last SLC images.",
    )
    out_group.add_argument(
        "-s",
        "--strides",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        default=(1, 1),
        help=(
            "Strides/decimation factor (x, y) (in pixels) to use when determining"
            " output shape."
        ),
    )
    out_group.add_argument(
        "--output-bounds",
        nargs=4,
        type=float,
        metavar=("left", "bottom", "right", "top"),
        help="Requested bounding box (in lat/lon) for final output.",
    )

    worker_group = parser.add_argument_group("Worker options")
    worker_group.add_argument(
        "--log-file", help="Path to log to, in addition to stderr"
    )
    worker_group.add_argument(
        "--work-directory",
        type=Path,
        default=Path().resolve(),
        help="Path to directory to store intermediate/output files.",
    )
    worker_group.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable usage of GPU (if using a machine that has one available).",
    )
    worker_group.add_argument(
        "--block-shape",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Shape (rows, col) of blocks of data to load at once time.",
    )
    worker_group.add_argument(
        "--n-parallel-bursts",
        type=int,
        default=1,
        help="Number of bursts to process in parallel.",
    )
    worker_group.add_argument(
        "--threads-per-worker",
        type=int,
        default=max(1, get_cpu_count() // 2),
        help="Number of threads to use per worker.",
    )
    parser.set_defaults(run_func=create_config)

    return parser
