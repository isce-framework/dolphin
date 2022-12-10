#!/usr/bin/env python
import argparse

from dolphin._log import log_runtime

from ._enums import WorkflowName
from .config import Config


@log_runtime
def run(config_file: str, debug: bool = False):
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.
    """
    cfg = Config.from_yaml(config_file)
    cfg.create_dir_tree(debug=debug)
    if cfg.workflow_name == "stack":
        from dolphin.workflows import s1_disp_stack

        s1_disp_stack.run(cfg, debug=debug)
    elif cfg.workflow_name == "single":
        raise NotImplementedError("Single interferogram workflow not yet implemented")
    else:
        choices = WorkflowName.__members__.values()
        raise ValueError(
            f"Unknown workflow name: {cfg.workflow_name}. Must be one of {choices}"
        )


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
    parser.set_defaults(run_func=run)
    return parser


def main(args=None):
    """Get the command line arguments and run the workflow."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    run(parsed_args.config_file, debug=parsed_args.debug)


if __name__ == "__main__":
    main()
