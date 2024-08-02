import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dolphin.filtering import filter_rasters

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def get_parser(subparser=None, subcommand_name="unwrap") -> argparse.ArgumentParser:
    """Set up the command line interface."""
    metadata = {
        "description": (
            "Filter unwrapped interferograms using a long-wavelength filter."
        ),
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
        type=Path,
        help=(
            "Path to output directory to store results. None stores in same location as"
            " inputs"
        ),
    )
    # Get Inputs from the command line
    inputs = parser.add_argument_group("Input options")
    inputs.add_argument(
        "--unw-filenames",
        nargs=argparse.ONE_OR_MORE,
        type=Path,
        help=(
            "List the paths of unwrapped files to filter. Can pass a newline delimited"
            " file with @ifg_filelist.txt"
        ),
    )
    inputs.add_argument(
        "--temporal-coherence-filename",
        type=Path,
        help="Optionally, list the path of the temporal coherence to mask.",
    )
    inputs.add_argument(
        "--cor-filenames",
        nargs=argparse.ZERO_OR_MORE,
        help="Optionally, list the paths of the correlation files to use for masking",
    )
    inputs.add_argument(
        "--conncomp-filenames",
        nargs=argparse.ZERO_OR_MORE,
        help="Optionally, list the paths of the connected component labels for masking",
    )
    parser.add_argument(
        "--wavelength-cutoff",
        type=float,
        default=50_000,
        help="Spatial wavelength_cutoff (in meters) of filter to use.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel files to filter.",
    )

    parser.set_defaults(run_func=_run_filter)

    return parser


def _run_filter(*args, **kwargs):
    """Run `dolphin.filtering.filter_long_wavelength`."""
    return filter_rasters(*args, **kwargs)


def main(args=None):
    """Get the command line arguments and filter files."""
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    return filter_rasters(**vars(parsed_args))


if __name__ == "__main__":
    main()
