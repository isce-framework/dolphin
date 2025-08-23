#!/usr/bin/env python3
"""Create boolean mask rasters from quality datasets for timeseries displacement data.

This script processes InSAR timeseries rasters and creates boolean masks based on
quality datasets (e.g., temporal coherence, similarity) that span date ranges.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache, reduce
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import tyro
from numpy.ma import MaskedArray
from opera_utils import get_dates

# Set up logging
logger = logging.getLogger("make_quality_masks")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _combine_quality_masks(
    quality_arrays: Sequence[MaskedArray],
    thresholds: Sequence[float],
    reduction_func: Callable[[MaskedArray, MaskedArray], MaskedArray],
) -> MaskedArray:
    """Threshold each *quality_array* then combine them with *reduction_func*."""
    if len(quality_arrays) != len(thresholds):
        msg = f"{len(quality_arrays)=} and {thresholds=} lengths must match."
        raise ValueError(msg)

    thresholded = (q > thr for q, thr in zip(quality_arrays, thresholds, strict=True))
    return reduce(reduction_func, thresholded)


@lru_cache(maxsize=6)
def _load_array(path: Path) -> MaskedArray:
    """Load a single-band raster to a NumPy array (cached)."""
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def main(
    timeseries_files: Sequence[Path | str],
    quality_patterns: Sequence[str],
    thresholds: Sequence[float],
    output_dir: Path | None = None,
    use_and: bool = False,
    overwrite: bool = False,
    output_suffix: str = "_mask",
) -> None:
    """Create boolean mask rasters for InSAR timeseries based on quality datasets.

    This script processes displacement timeseries rasters and creates boolean masks
    by combining multiple quality datasets (e.g., temporal coherence, similarity)
    that span different date ranges.

    Parameters
    ----------
    timeseries_files : Sequence[Path | str]
        Sequence of timeseries raster file paths to process.
    quality_patterns : Sequence[str]
        Glob patterns for finding quality raster files.
        Example: ['interferograms/sim*.tif', 'interferograms/temporal_coherence*.tif']
    thresholds : Sequence[float]
        Thresholds for each quality pattern (same order as quality_patterns).
        Example: [0.4, 0.6]
    output_dir : Path | None, optional
        Output directory for mask files.
        If None, defaults to directory of first timeseries file.
    use_and : bool, optional
        If True, use logical AND to combine quality masks (all must pass).
        If False, use logical OR (any can pass). Defaults to False.
    overwrite : bool, optional
        If True, overwrite existing mask files. Defaults to False.
    output_suffix : str, optional
        Suffix for output mask files. Defaults to "_mask".

    """
    if len(quality_patterns) != len(thresholds):
        msg = "Number of quality patterns must match number of thresholds"
        raise ValueError(msg)

    # Find all quality files for each pattern
    all_quality_files: list[Path] = []
    quality_file_groups: list[list[Path]] = []
    for pattern in quality_patterns:
        quality_files = sorted(map(Path, glob(pattern)))
        all_quality_files.extend(quality_files)
        quality_file_groups.append(quality_files)

    if not all_quality_files:
        msg = "No quality files found for any pattern"
        raise ValueError(msg)
    logger.info(f"Processing {len(timeseries_files)} timeseries files")
    logger.info(f"Found {len(all_quality_files)} quality files")

    if output_dir is None:
        output_dir = Path(timeseries_files[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    reduction = np.logical_and if use_and else np.logical_or

    date_to_quality_files: Mapping[tuple[datetime, datetime], list[Path]] = defaultdict(
        list
    )
    date_to_timeseries_files: Mapping[tuple[datetime, datetime], list[Path]] = (
        defaultdict(list)
    )
    for f in all_quality_files:
        d_start, d_end = tuple(get_dates(f)[:2])
        cur_timeseries_files = [
            Path(p) for p in timeseries_files if d_start <= get_dates(p)[1] <= d_end
        ]
        date_to_quality_files[(d_start, d_end)].append(f)
        date_to_timeseries_files[(d_start, d_end)].extend(cur_timeseries_files)

    with rasterio.open(timeseries_files[0]) as src:
        profile = src.profile.copy()

    for date_tup, q_paths in date_to_quality_files.items():
        if not q_paths:
            logger.warning("No quality raster covers %s - skipping", date_tup)
            continue

        logger.info(f"{date_tup} quality files:")
        logger.info(f"{q_paths}")
        # make the current quality mask (applies to a list of time series files)
        q_arrays: list[MaskedArray] = [_load_array(p) for p in q_paths]
        mask = _combine_quality_masks(q_arrays, thresholds, reduction)

        cur_files = date_to_timeseries_files[date_tup]
        for ts_path in cur_files:
            out_name = f"{Path(ts_path).stem}{output_suffix}.tif"
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                logger.info("Mask exists - skipping %s", out_path.name)
                continue

            nodata = 255
            dtype = "uint8"
            prof = profile.copy() | {"dtype": dtype, "nodata": nodata}
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(mask.filled(nodata).astype(dtype), 1)

            logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    tyro.cli(main)
