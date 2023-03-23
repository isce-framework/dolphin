import argparse

import dolphin.workflows._config_cli
import dolphin.workflows._run_cli
import dolphin.workflows._validate_cli
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
    dolphin.workflows._run_cli.get_parser(subparser, "run")
    dolphin.workflows._config_cli.get_parser(subparser, "config")
    dolphin.workflows._validate_cli.get_parser(subparser, "validate")
    parsed_args = parser.parse_args(args=args)

    arg_dict = vars(parsed_args)
    run_func = arg_dict.pop("run_func")
    run_func(**arg_dict)
