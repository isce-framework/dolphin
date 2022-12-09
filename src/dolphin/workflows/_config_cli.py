#!/usr/bin/env python
import argparse

from ._enums import WorkflowName
from .config import Config


def create_config(
    outfile,
    *,
    cslc_directory=None,
    cslc_file_ext=".nc",
    cslc_file_list=None,
    mask_files=None,
):
    """Create a config for a displacement workflow."""
    cfg = Config(
        inputs={
            "cslc_directory": cslc_directory,
            "cslc_file_ext": cslc_file_ext,
            "cslc_file_list": cslc_file_list,
            "mask_files": mask_files,
        }
    )
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
        help="Name of YAML configuration file to save to.",
    )
    # Get Inputs from the command line
    parser.add_argument(
        "--cslc-directory",
        help="Path to directory containing the SLCs.",
    )
    parser.add_argument(
        "--cslc-file",
        nargs=argparse.ZERO_OR_MORE,
        help="Path to a file containing a list of SLCs.",
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

    return parser


def main(args=None):
    """Get the command line arguments and create the config file."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    create_config(**vars(parsed_args))


if __name__ == "__main__":
    main()
