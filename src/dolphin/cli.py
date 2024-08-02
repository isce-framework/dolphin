import argparse
import sys

import dolphin._cli_filter
import dolphin._cli_timeseries
import dolphin._cli_unwrap
import dolphin.workflows._cli_config
import dolphin.workflows._cli_run
from dolphin import __version__


def main(args=None):
    """Top-level command line interface to the workflows."""
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparser = parser.add_subparsers(title="subcommands")
    parser.add_argument("--version", action="version", version=__version__)

    # Adds the subcommand to the top-level parser
    dolphin.workflows._cli_run.get_parser(subparser, "run")
    dolphin.workflows._cli_config.get_parser(subparser, "config")
    dolphin._cli_unwrap.get_parser(subparser, "unwrap")
    dolphin._cli_timeseries.get_parser(subparser, "timeseries")
    dolphin._cli_filter.get_parser(subparser, "filter")
    parsed_args = parser.parse_args(args=args)

    arg_dict = vars(parsed_args)
    try:
        run_func = arg_dict.pop("run_func")
        run_func(**arg_dict)
    except KeyError:
        # No arguments passed
        parser.print_help()
        sys.exit(1)
