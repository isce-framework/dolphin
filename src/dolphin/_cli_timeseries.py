import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dolphin.workflows import CallFunc

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def get_parser(subparser=None, subcommand_name="timeseries") -> argparse.ArgumentParser:
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
        "--output-dir",
        default=Path(),
        help="Path to output directory to store results",
    )
    parser.add_argument(
        "--unwrapped-paths",
        nargs=argparse.ZERO_OR_MORE,
        help=(
            "List the paths of all unwrapped interferograms. Can pass a "
            "newline delimited file with @ifg_filelist.txt"
        ),
    )
    parser.add_argument(
        "--conncomp-paths",
        nargs=argparse.ZERO_OR_MORE,
        help=(
            "List the paths of all connected component files. Can pass a "
            "newline delimited file with @conncomp_filelist.txt"
        ),
    )
    parser.add_argument(
        "--corr-paths",
        nargs=argparse.ZERO_OR_MORE,
        help=(
            "List the paths of all correlation files. Can pass a newline delimited"
            " file with @cor_filelist.txt"
        ),
    )
    parser.add_argument(
        "--condition-file",
        help=(
            "A file with the same size as each raster, like amplitude dispersion or"
            "temporal coherence to find reference point. default: amplitude dispersion"
        ),
    )
    parser.add_argument(
        "--condition",
        type=CallFunc,
        default=CallFunc.MIN,
        help=(
            "A condition to apply to condition file to find the reference point"
            "Options are [min, max]. default=min"
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=5,
        help="Number of threads for the inversion",
    )
    parser.add_argument(
        "--run-velocity",
        action="store_true",
        help="Run the velocity estimation from the phase time series",
    )
    parser.add_argument(
        "--reference-point",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=(-1, -1),
        help=(
            "Reference point (row, col) used if performing a time series inversion. "
            "If not provided, a point will be selected from a consistent connected "
            "component with low amplitude dispersion or high temporal coherence."
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=range_limited_float_type,
        default=0.2,
        metavar="[0-1]",
        help="Pixels with correlation below this value will be masked out.",
    )

    parser.set_defaults(run_func=_run_timeseries)

    return parser


def range_limited_float_type(arg):
    """Type function for argparse - a float within some predefined bounds."""
    try:
        f = float(arg)
    except ValueError as err:
        raise argparse.ArgumentTypeError("Must be a floating point number") from err
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError(
            "Argument must be < " + str(1) + "and > " + str(0)
        )
    return f


def _run_timeseries(*args, **kwargs):
    """Run `dolphin.timeseries.run`.

    Wrapper for the dolphin.timeseries to invert and create velocity.
    """
    from dolphin import timeseries

    return timeseries.run(*args, **kwargs)


def main(args=None):
    """Get the command line arguments for timeseries inversion."""
    from dolphin import timeseries

    parser = get_parser()
    parsed_args = parser.parse_args(args)

    timeseries.run(**vars(parsed_args))


if __name__ == "__main__":
    main()
