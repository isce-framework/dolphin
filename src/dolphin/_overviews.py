import argparse
import logging
from enum import Enum
from os import fspath
from pathlib import Path
from typing import Sequence

from osgeo import gdal
from tqdm.contrib.concurrent import thread_map

from dolphin._types import PathOrStr

gdal.UseExceptions()

logger = logging.getLogger(__name__)

raster_type_to_resampling = {
    "unw": "average",
}


class Resampling(Enum):
    """GDAL resampling algorithm."""

    NEAREST = "nearest"
    AVERAGE = "average"
    LANCZOS = "lanczos"


def add_overviews(
    file_path: Path | str,
    overview_levels: Sequence[int] = [4, 8, 16, 32, 64],
    resampling: Resampling = Resampling.NEAREST,
    external: bool = False,
    compression: str = "LZW",
):
    """Add GDAL compressed overviews to an existing file.

    Parameters
    ----------
    file_path : Path
        Path to the file to process.
    overview_levels : list
        List of overview levels to add.
    resampling : str or Resampling
        GDAL resampling algorithm for overviews.
        Default = "nearest"
    external : bool, default = False
        Use external overviews (.ovr files).
    compression: str, default = "LZW"
        Compression algorithm to use for overviews.
        See https://gdal.org/programs/gdaladdo.html for options.

    """
    flags = gdal.GA_Update if not external else gdal.GA_ReadOnly
    ds = gdal.Open(fspath(file_path), flags)
    if ds.GetRasterBand(1).GetOverviewCount() > 0:
        logger.info("%s already has overviews. Skipping.")
        return

    gdal.SetConfigOption("COMPRESS_OVERVIEW", compression)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "2")
    ds.BuildOverviews(resampling.value, overview_levels)


def process_files(
    file_paths: Sequence[PathOrStr],
    levels: Sequence[int],
    resampling: Resampling,
    max_workers: int = 5,
) -> None:
    """Process files to add GDAL overviews and compression.

    Parameters
    ----------
    file_paths : Sequence[PathOrStr]
        Sequence of file paths to process.
    levels : Sequence[int]
        Sequence of overview levels to add.
    resampling : str or Resampling
        GDAL resampling algorithm for overviews.
        Default = "nearest"
    max_workers : int, default = 5
        Number of parallel threads to run.

    """
    thread_map(
        lambda file_path: add_overviews(
            Path(file_path),
            overview_levels=list(levels),
            resampling=resampling,
        ),
        file_paths,
        max_workers=max_workers,
    )


def run():
    """Add compressed GDAL overviews to files."""
    parser = argparse.ArgumentParser(
        description="Add compressed GDAL overviews to files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_paths", nargs="+", type=str, help="Path to files")
    parser.add_argument(
        "--levels",
        "-l",
        nargs="*",
        default=[4, 8, 16, 32, 64],
        type=int,
        help="Overview levels to add.",
    )
    parser.add_argument(
        "--resampling",
        "-r",
        default=Resampling("nearest"),
        choices=[r.value for r in Resampling],
        type=Resampling,
        help="Resampling algorithm to use when building overviews",
    )
    parser.add_argument(
        "--max-workers",
        "-n",
        default=5,
        type=int,
        help="Number of parallel files to process",
    )

    args = parser.parse_args()

    # Convert resampling argument from string to Resampling Enum
    resampling_enum = Resampling(args.resampling)

    process_files(
        file_paths=args.file_paths,
        levels=args.levels,
        resampling=resampling_enum,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
