import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dolphin.filtering import filter_long_wavelength

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
        "--output-dir",
        type=Path,
        help="Path to output directory to store results. None stores in same location as inputs",
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
    temporal_coherence_filename: Path | None,
    wavelength_cutoff: float = 50_000,
    output_dir: Path | None = None,
    num_threads: int = 4,
    **kwargs,
):
    """Recreate and compress a list of raster files.

    Useful for rasters which were created in block and lost
    the full effect of compression.

    Parameters
    ----------
    raster_files : List[Path]
        List of paths to the input raster files.
    output_dir : Path, optional
        Directory to save the processed rasters or None for in-place processing.
    num_threads : int, optional
        Number of threads to use (default is 4).
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to None.
        Lower numbers will truncate the mantissa more and enable more compression.

    Returns
    -------
    output_dir : Path
        Path to newly created file.
        If `output_dir` is None, this will be the same as `raster_paths`

    """
    import multiprocessing as mp
    from itertools import repeat

    import numpy as np
    from tqdm.contrib.concurrent import process_map

    from dolphin import io

    if temporal_coherence_filename:
        bad_pixel_mask = io.load_gdal(temporal_coherence_filename) < 0.6
    else:
        bad_pixel_mask = np.zeros(
            io.get_raster_xysize(unw_filenames[0])[::-1], dtype="bool"
        )

    if output_dir is None:
        assert unw_filenames
        output_dir = unw_filenames[0].parent

    mp.set_start_method("spawn")

    process_map(
        _filter_and_save,
        unw_filenames,
        # cor_paths,
        # conncomp_paths,
        repeat(output_dir),
        repeat(wavelength_cutoff),
        repeat(bad_pixel_mask),
        max_workers=num_threads,
        desc="Filtering rasters",
    )


def _filter_and_save(
    unw_filename: Path, output_dir: Path, wavelength_cutoff: float, mask
) -> Path:
    from dolphin import io

    unw = io.load_gdal(unw_filename)
    filt_arr = filter_long_wavelength(
        unwrapped_phase=unw, wavelength_cutoff=wavelength_cutoff, bad_pixel_mask=mask
    )
    io.round_mantissa(filt_arr, keep_bits=9)
    output_name = output_dir / Path(unw_filename).with_suffix(".filt.tif").name
    io.write_arr(arr=filt_arr, like_filename=unw_filename, output_name=output_name)
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
