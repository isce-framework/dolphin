import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dolphin._log import setup_logging
from dolphin.timeseries import InversionMethod

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def get_parser(subparser=None, subcommand_name="timeseries") -> argparse.ArgumentParser:
    """Set up the command line interface."""
    metadata = {
        "description": "Create a configuration file for a displacement workflow.",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
        "fromfile_prefix_chars": "@",
    }
    if subparser:
        parser = subparser.add_parser(subcommand_name, **metadata)
    else:
        parser = argparse.ArgumentParser(**metadata)  # type: ignore[arg-type]

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
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
        "--quality-file",
        help=(
            "A file with the same size as each raster, like amplitude dispersion or "
            "temporal coherence to find reference point"
        ),
    )
    parser.add_argument(
        "--method",
        type=InversionMethod,
        choices=list(InversionMethod),
        default=InversionMethod.L1,
        help=(
            "Inversion method to use when solving Ax = b. L2 uses least squares"
            " (faster), L1 minimizes |Ax - b|_1"
        ),
    )
    parser.add_argument(
        "--run-velocity",
        action="store_true",
        help="Run the velocity estimation from the phase time series",
    )
    parser.add_argument(
        "--weight-velocity-by-corr",
        action="store_true",
        help=(
            "Flag to indicate whether the velocity fitting should use correlation as"
            " weights"
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=range_limited_float_type,
        default=0.0,
        metavar="[0-1]",
        help="Pixels with correlation below this value will be masked out",
    )
    parser.add_argument(
        "--block-shape",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("HEIGHT", "WIDTH"),
        help="The shape of the blocks to process in parallel",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="The parallel blocks to process at once",
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
            "component with low amplitude dispersion or high temporal coherence"
        ),
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        help=(
            "The wavelength of the radar signal, in meters. If provided, the output "
            "rasters are in meters and meters/year for displacement and velocity. "
            "If not provided, outputs are in radians"
        ),
    )
    parser.add_argument(
        "--add-overviews",
        action="store_true",
        default=True,
        help="If True, creates overviews of the new velocity raster",
    )
    parser.add_argument(
        "--extra-reference-date",
        type=lambda s: datetime.strptime(s, "%Y%m%d"),
        help=(
            "If provided, makes another set of interferograms referenced to this "
            "for all dates later than it. Format: YYYYMMDD"
        ),
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

    setup_logging()
    return timeseries.run(*args, **kwargs)


def main(args=None):
    """Get the command line arguments for timeseries inversion."""
    from dolphin import timeseries

    setup_logging()
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    timeseries.run(**vars(parsed_args))


if __name__ == "__main__":
    main()
