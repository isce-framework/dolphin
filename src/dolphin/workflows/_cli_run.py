from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def run(
    config_file: str,
    debug: bool = False,
) -> None:
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    # rest of imports here so --help doesn't take forever

    from . import displacement
    from .config import DisplacementWorkflow

    cfg = DisplacementWorkflow.from_yaml(config_file)
    displacement.run(cfg, debug=debug)


def get_parser(
    subparser: Optional[_SubparserType] = None, subcommand_name: str = "run"
) -> argparse.ArgumentParser:
    """Set up the command line interface."""
    metadata = {
        "description": "Run a displacement workflow",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
    }
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)  # type: ignore[arg-type]

    else:
        parser = argparse.ArgumentParser(**metadata)  # type: ignore[arg-type]

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


def main(args: Optional[Sequence[str]] = None) -> None:
    """Get the command line arguments and run the workflow."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    run(parsed_args.config_file, debug=parsed_args.debug)


if __name__ == "__main__":
    main()
