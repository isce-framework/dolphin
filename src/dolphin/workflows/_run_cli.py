#!/usr/bin/env python
import argparse
from typing import Optional

from dolphin._log import get_log, log_runtime


@log_runtime
def run(config_file: str, debug: bool = False, log_file: Optional[str] = None):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.
    log_file : str, optional
        If provided, will log to this file in addition to stderr.
    """
    from threadpoolctl import ThreadpoolController

    from . import s1_disp
    from .config import Workflow

    # Set the logging level for all `dolphin.` modules
    get_log("dolphin", debug=debug)

    # Set the environment variables for the workers
    # TODO: Is this the best place to do this?
    cfg = Workflow.from_yaml(config_file)
    cfg.create_dir_tree(debug=debug)

    controller = ThreadpoolController()
    controller.limit(limits=cfg.worker_settings.threads_per_worker)

    s1_disp.run(cfg, debug=debug, log_file=log_file)


def get_parser(subparser=None, subcommand_name="run"):
    """Set up the command line interface."""
    metadata = dict(
        description="Run a displacement workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)
    else:
        parser = argparse.ArgumentParser(**metadata)

    parser.add_argument(
        "config_file",
        help="Name of YAML configuration file describing workflow options.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages to the log.",
    )
    parser.add_argument(
        "--log-file",
        help="If provided, will log to this file in addition to stderr.",
    )
    parser.set_defaults(run_func=run)
    return parser


def main(args=None):
    """Get the command line arguments and run the workflow."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    run(parsed_args.config_file, debug=parsed_args.debug)


if __name__ == "__main__":
    main()
