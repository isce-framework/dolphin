#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ._enums import WorkflowName
from .config import Workflow


def create_config(
    *,
    outfile: Union[str, Path],
    slc_directory=None,
    ext: str = ".nc",
    slc_files: Optional[List[str]] = None,
    mask_files: Optional[List[str]] = None,
    strides: Tuple[int, int],
    name: WorkflowName = WorkflowName.STACK,
):
    """Create a config for a displacement workflow."""
    cfg = Workflow(
        inputs=dict(
            cslc_directory=slc_directory,
            cslc_file_ext=ext,
            cslc_file_list=slc_files,
            mask_files=mask_files,
        ),
        outputs=dict(
            strides={"x": strides[0], "y": strides[1]},
        ),
        name=name,
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
    )
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)
    else:
        parser = argparse.ArgumentParser(**metadata)

    parser.add_argument(
        "-o",
        "--outfile",
        default="dolphin_config.yaml",
        help="Name of YAML configuration file to save to. Use '-' to write to stdout.",
    )
    # Get Inputs from the command line
    parser.add_argument(
        "-d",
        "--slc-directory",
        help="Path to directory containing the SLCs.",
    )
    parser.add_argument(
        "--ext",
        default=".nc",
        help="Extension of SLCs to search for (if --slc-directory is given).",
    )
    parser.add_argument(
        "--slc-files",
        nargs=argparse.ZERO_OR_MORE,
        help="Alternative: list the paths of all SLC files to include.",
    )
    # Get Outputs from the command line
    parser.add_argument(
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
    parser.add_argument(
        "-n",
        "--name",
        type=str.lower,
        choices=[i.name.lower() for i in WorkflowName],
        default=WorkflowName.STACK.value,
        help="Name of the displacement workflow. YAML configuration file to save to.",
    )
    parser.add_argument(
        "--mask-files",
        nargs=argparse.ZERO_OR_MORE,
        help="Path to a file containing a list of mask files.",
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
