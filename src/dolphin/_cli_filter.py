import argparse
import multiprocessing as mp
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from tqdm.contrib.concurrent import process_map

from dolphin import io
from dolphin._overviews import Resampling, create_image_overviews
from dolphin.filtering import filter_long_wavelength

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
        "--max-jobs",
        type=int,
        default=1,
        help="Number of parallel files to filter.",
    )

    parser.set_defaults(run_func=_run_filter)

    return parser


def filter_rasters(
    unw_filenames: list[Path],
    cor_filenames: list[Path] | None,
    conncomp_filenames: list[Path] | None,
    temporal_coherence_filename: Path | None,
    wavelength_cutoff: float = 50_000,
    correlation_cutoff: float = 0.5,
    output_dir: Path | None = None,
    num_threads: int = 4,
):
    """Filter a list of unwrapped interferogram files using a long-wavelength filter.

    This function applies a spatial filter to remove long-wavelength components from
    unwrapped interferograms. It can optionall use temporal coherence, correlation, and
    connected component information for masking.

    Parameters
    ----------
    unw_filenames : list[Path]
        List of paths to unwrapped interferogram files to be filtered.
    cor_filenames : list[Path] | None
        List of paths to correlation files
        Passing None skips filtering on correlation.
    conncomp_filenames : list[Path] | None
        List of paths to connected component files, filters any 0 labelled pixels.
        Passing None skips filtering on connected component labels.
    temporal_coherence_filename : Path | None
        Path to the temporal coherence file for masking.
        Passing None skips filtering on temporal coherence.
    wavelength_cutoff : float, optional
        Spatial wavelength cutoff (in meters) for the filter. Default is 50,000 meters.
    correlation_cutoff : float, optional
        Threshold of correlation (if passing `cor_filenames`) to use to ignore pixels
        during filtering.
    output_dir : Path | None, optional
        Directory to save the filtered results.
        If None, saves in the same location as inputs with .filt.tif extension.
    num_threads : int, optional
        Number of parallel threads to use for processing. Default is 4.

    Notes
    -----
    - If temporal_coherence_filename is provided, pixels with coherence < 0.5 are masked

    """
    bad_pixel_mask = np.zeros(
        io.get_raster_xysize(unw_filenames[0])[::-1], dtype="bool"
    )
    if temporal_coherence_filename:
        bad_pixel_mask = bad_pixel_mask | (
            io.load_gdal(temporal_coherence_filename) < 0.5
        )

    if output_dir is None:
        assert unw_filenames
        output_dir = unw_filenames[0].parent
    output_dir.mkdir(exist_ok=True)

    mp.set_start_method("spawn")

    process_map(
        _filter_and_save,
        unw_filenames,
        cor_filenames or repeat(None),
        conncomp_filenames or repeat(None),
        repeat(output_dir),
        repeat(wavelength_cutoff),
        repeat(bad_pixel_mask),
        repeat(correlation_cutoff),
        max_workers=num_threads,
        desc="Filtering rasters",
    )


def _get_pixel_spacing(filename):
    _, x_res, _, _, _, y_res = io.get_raster_gt(filename)
    return (abs(x_res) + abs(y_res)) / 2


def _filter_and_save(
    unw_filename: Path,
    cor_path: Path | None,
    conncomp_path: Path | None,
    output_dir: Path,
    wavelength_cutoff: float,
    bad_pixel_mask: NDArray[np.bool_],
    correlation_cutoff: float = 0.5,
) -> Path:
    # Average for the pixel spacing for filtering
    pixel_spacing = _get_pixel_spacing(unw_filename)

    if cor_path is not None:
        bad_pixel_mask |= io.load_gdal(cor_path) < correlation_cutoff
    if conncomp_path is not None:
        bad_pixel_mask |= io.load_gdal(conncomp_path, masked=True).astype(bool) == 0

    unw = io.load_gdal(unw_filename)
    filt_arr = filter_long_wavelength(
        unwrapped_phase=unw,
        wavelength_cutoff=wavelength_cutoff,
        bad_pixel_mask=bad_pixel_mask,
        pixel_spacing=pixel_spacing,
        workers=1,
    )
    io.round_mantissa(filt_arr, keep_bits=9)
    output_name = output_dir / Path(unw_filename).with_suffix(".filt.tif").name
    io.write_arr(arr=filt_arr, like_filename=unw_filename, output_name=output_name)

    create_image_overviews(output_name, resampling=Resampling.AVERAGE)

    return output_name


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
