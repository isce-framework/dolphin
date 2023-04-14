#!/usr/bin/env python
import argparse
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .config import OPERA_DATASET_NAME, InterferogramNetworkType, Workflow, WorkflowName


def create_config(
    *,
    outfile: Union[str, Path],
    slc_files: Optional[List[str]] = None,
    subdataset: Optional[str] = None,
    mask_file: Optional[str] = None,
    ministack_size: Optional[int] = 15,
    amp_dispersion_threshold: float = 0.25,
    strides: Tuple[int, int],
    block_size_gb: float = 1,
    n_workers: int = 16,
    threads_per_worker: int = 1,
    no_gpu: bool = False,
    use_icu: bool = False,
    single_update: bool = False,
    log_file: Optional[Path] = None,
):
    """Create a config for a displacement workflow."""
    if single_update:
        # create only one interferogram from the first and last SLC images
        interferogram_network = dict(
            network_type=InterferogramNetworkType.MANUAL_INDEX,
            indexes=[(0, -1)],
        )
        workflow_name = WorkflowName.SINGLE
        # Override the ministack size so that only one phase linking is run
        ministack_size = 1000
    else:
        interferogram_network = {}  # Use default
        workflow_name = WorkflowName.STACK

    cfg = Workflow(
        workflow_name=workflow_name,
        cslc_file_list=slc_files,
        mask_file=mask_file,
        input_options=dict(
            subdataset=subdataset,
        ),
        interferogram_network=interferogram_network,
        output_options=dict(
            strides={"x": strides[0], "y": strides[1]},
        ),
        phase_linking=dict(
            ministack_size=ministack_size,
        ),
        ps_options=dict(
            amp_dispersion_threshold=amp_dispersion_threshold,
        ),
        unwrap_options=dict(
            unwrap_method=("icu" if use_icu else "snaphu"),
        ),
        worker_settings=dict(
            block_size_gb=block_size_gb,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            gpu_enabled=(not no_gpu),
        ),
        log_file=log_file,
    )

    if outfile == "-":  # Write to stdout
        cfg.to_yaml(sys.stdout)
    else:
        print(f"Saving configuration to {str(outfile)}", file=sys.stderr)
        cfg.to_yaml(outfile)


def get_parser(subparser=None, subcommand_name="run"):
    """Set up the command line interface."""
    metadata = dict(
        description="Create a configuration file for a displacement workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # https://docs.python.org/3/library/argparse.html#fromfile-prefix-chars
        fromfile_prefix_chars="@",
    )
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)
    else:
        parser = argparse.ArgumentParser(**metadata)

    # parser._action_groups.pop()
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
        help="Alternative: list the paths of all SLC files to include.",
    )

    # Get the subdataset of the SLCs to use, if passing HDF5/NetCDF files
    inputs.add_argument(
        "-sds",
        "--subdataset",
        help=(
            "Subdataset to use from HDF5/NetCDF files. For OPERA CSLC NetCDF files, if"
            f" None is passed, the default is {OPERA_DATASET_NAME}."
        ),
    )
    parser.add_argument(
        "--mask-file",
        help=(
            "Path to Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data."
        ),
    )
    parser.add_argument("--log-file", help="Path to log to, in addition to stderr")

    # Phase linking options
    pl_group = parser.add_argument_group("Phase Linking options")
    pl_group.add_argument(
        "-ms",
        "--ministack-size",
        default=15,
        help="Strides/decimation factor (x, y) (in pixels) to use when determining",
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
        "--use-icu",
        action="store_true",
        help="Use the ICU algorithm instead of the default SNAPHU.",
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

    worker_group = parser.add_argument_group("Worker options")
    worker_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable the GPU (if using a machine that has one available).",
    )
    worker_group.add_argument(
        "--block-size-gb",
        type=float,
        default=1,
        help="Size (in GB) of blocks of data to load at once time.",
    )
    worker_group.add_argument(
        "--n-workers",
        type=int,
        default=cpu_count(),
        help="Number of CPU workers to use (for CPU processing).",
    )
    worker_group.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Number of threads to use per worker.",
    )
    parser.set_defaults(run_func=create_config)

    return parser


def main(args=None):
    """Get the command line arguments and create the config file."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    create_config(**vars(parsed_args))


if __name__ == "__main__":
    main()
