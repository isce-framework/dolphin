from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dolphin.workflows.config import UnwrapMethod

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def get_parser(subparser=None, subcommand_name="unwrap") -> argparse.ArgumentParser:
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
        parser = argparse.ArgumentParser(**metadata)  # type: ignore[arg-type]

    # parser._action_groups.pop()
    parser.add_argument(
        "-o",
        "--output-path",
        default=Path(),
        help="Path to output directory to store results",
    )
    # Get Inputs from the command line
    inputs = parser.add_argument_group("Input options")
    inputs.add_argument(
        "--ifg-filenames",
        nargs=argparse.ZERO_OR_MORE,
        help=(
            "List the paths of all ifg files to include. Can pass a newline delimited"
            " file with @ifg_filelist.txt"
        ),
    )
    inputs.add_argument(
        "--cor-filenames",
        nargs=argparse.ZERO_OR_MORE,
        help=(
            "List the paths of all ifg files to include. Can pass a newline delimited"
            " file with @cor_filelist.txt"
        ),
    )
    inputs.add_argument(
        "--mask-filename",
        help=(
            "Path to Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data."
        ),
    )
    inputs.add_argument(
        "--temp-coh-filename",
        help="Path to temporal coherence file from phase linking",
    )
    parser.add_argument(
        "--nlooks",
        type=int,
        help="Effective number of looks used to form correlation",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Number of parallel files to unwrap",
    )

    algorithm_opts = parser.add_argument_group("Algorithm options")
    algorithm_opts.add_argument(
        "--unwrap-method",
        type=UnwrapMethod,
        choices=[m.value for m in UnwrapMethod],
        default=UnwrapMethod.SNAPHU.value,
        help="Choice of unwrapping algorithm to use.",
    )
    algorithm_opts.add_argument(
        "--run-goldstein",
        action="store_true",
        help="Run Goldstein filter before unwrapping.",
    )
    algorithm_opts.add_argument(
        "--run-interpolation",
        action="store_true",
        help="Run interpolation before unwrapping.",
    )

    spurt_opts = parser.add_argument_group("Spurt options")
    spurt_opts.add_argument(
        "--temp-coh-threshold",
        type=float,
        help="Cutoff on temporal_coherence raster to choose pixels for unwrapping.",
    )

    tophu_opts = parser.add_argument_group("Tophu options")
    # Add ability for downsampling/tiling with tophu
    tophu_opts.add_argument(
        "--ntiles",
        type=int,
        nargs=2,
        metavar=("ROW_TILES", "COL_TILES"),
        default=(1, 1),
        help=(
            "(using tophu) Split the interferograms into this number of tiles along the"
            " (row, col) axis."
        ),
    )
    tophu_opts.add_argument(
        "--downsample-factor",
        type=int,
        nargs=2,
        default=(1, 1),
        help=(
            "(using tophu) Downsample the interferograms by this factor "
            " during multiresolution unwrapping."
        ),
    )
    parser.set_defaults(run_func=_run_unwrap)

    return parser


def _run_unwrap(*args, **kwargs):
    """Run `dolphin.unwrap.run`.

    Wrapper for the dolphin.unwrap to delay import time.
    """
    from dolphin import unwrap

    return unwrap.run(*args, **kwargs)


def main(args=None):
    """Get the command line arguments and unwrap files."""
    from dolphin import unwrap

    parser = get_parser()
    parsed_args = parser.parse_args(args)
    unwrap.run(**vars(parsed_args))


if __name__ == "__main__":
    main()
