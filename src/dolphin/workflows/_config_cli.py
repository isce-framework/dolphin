#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .config import OPERA_DATASET_NAME, InterferogramNetworkType, Workflow


def create_config(
    *,
    outfile: Union[str, Path],
    slc_files: Optional[List[str]] = None,
    subdataset: Optional[str] = None,
    mask_files: Optional[List[str]] = None,
    ministack_size: Optional[int] = 15,
    strides: Tuple[int, int],
    max_ram_gb: float = 1,
    n_workers: int = 16,
    no_gpu: bool = False,
    single_update: bool = False,
):
    """Create a config for a displacement workflow."""
    if single_update:
        # create only one interferogram from the first and last SLC images
        interferogram_network = dict(
            network_type=InterferogramNetworkType.MANUAL_INDEX,
            indexes=[(0, -1)],
        )
        # Override the ministack size so that only one phase linking is run
        ministack_size = 1000
    else:
        interferogram_network = None  # Use default

    cfg = Workflow(
        inputs=dict(
            cslc_file_list=slc_files,
            mask_files=mask_files,
            subdataset=subdataset,
        ),
        interferogram_network=interferogram_network,
        outputs=dict(
            strides={"x": strides[0], "y": strides[1]},
        ),
        phase_linking=dict(
            ministack_size=ministack_size,
        ),
        worker_settings=dict(
            max_ram_gb=max_ram_gb,
            n_workers=n_workers,
            gpu_enabled=(not no_gpu),
        ),
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

    # Phase linking options
    pl_group = parser.add_argument_group("Phase Linking options")
    pl_group.add_argument(
        "-ms",
        "--ministack-size",
        default=15,
        help="Strides/decimation factor (x, y) (in pixels) to use when determining",
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
        "--max-ram-gb",
        type=float,
        default=1,
        help="Maximum amount of RAM to use per worker.",
    )
    worker_group.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of workers to use.",
    )
    # parser.add_argument(
    #     "--mask-files",
    #     nargs=argparse.ZERO_OR_MORE,
    #     help="Path to a file containing a list of mask files.",
    # )
    parser.set_defaults(run_func=create_config)

    return parser


def main(args=None):
    """Get the command line arguments and create the config file."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    create_config(**vars(parsed_args))


if __name__ == "__main__":
    main()
