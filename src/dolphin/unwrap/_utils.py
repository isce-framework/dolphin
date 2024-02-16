from __future__ import annotations

from os import fspath
from pathlib import Path

import rasterio as rio

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)


def create_combined_mask(
    mask_filename: Filename,
    image_filename: Filename,
    output_filename: Filename | None = None,
    band: int = 1,
) -> Path:
    """Create a combined a nodata mask from `image_filename` with `mask_filename`.

    This function reads the nodata values from `image_filename` and the existing
    mask from `mask_filename`. It combines these two masks such that a pixel is
    considered valid only if it is valid in both masks.

    Parameters
    ----------
    mask_filename : PathOrStr
        The file path of the existing mask file. This mask specifies pixels that
        are valid (1) or invalid (0).
    image_filename : PathOrStr
        The file path of the image file. This file's nodata values are used to
        generate a mask of valid and invalid pixels.
    output_filename : PathOrStr, optional, default=None
        The file path where the combined mask will be saved.
        If None, creates "combined_mask.tif" in the same directory as `mask_filename`
    band : int, default=1
        The band of the image from which to read the nodata values.

    Returns
    -------
    Path
        The path to the created combined mask file.

    Raises
    ------
    ValueError
        If `image_filename` does not have any nodata values defined

    """
    with rio.open(image_filename) as src:
        if not src.nodatavals:
            msg = f"{image_filename} does not have any `nodata` values."
            raise ValueError(msg)
        # https://rasterio.readthedocs.io/en/stable/topics/masks.html
        # rasterio loads this where 0 means the pixel is masked, and
        # 255 is valid.
        # coerce to True == valid pixels
        nd_mask = src.read_masks(band).astype(bool)

    # In the existing mask, 1 should be valid, 0 invalid
    mask = io.load_gdal(mask_filename).astype(bool)
    # A valid output has to be valid in the mask, AND not be a `nodata`
    combined = mask & nd_mask

    if output_filename is None:
        output_filename = Path(mask_filename).parent / "combined_mask.tif"
    io.write_arr(like_filename=mask_filename, arr=combined, output_name=output_filename)
    return Path(output_filename)


def _zero_from_mask(
    ifg_filename: Filename, corr_filename: Filename, mask_filename: Filename
) -> tuple[Path, Path]:
    zeroed_ifg_file = Path(ifg_filename).with_suffix(".zeroed.tif")
    zeroed_corr_file = Path(corr_filename).with_suffix(".zeroed.cor.tif")

    if io.get_raster_xysize(ifg_filename) != io.get_raster_xysize(mask_filename):
        msg = f"Mask {mask_filename} and {ifg_filename} shapes don't match"
        raise ValueError(msg)

    mask = io.load_gdal(mask_filename)
    for in_f, out_f in zip(
        [ifg_filename, corr_filename], [zeroed_ifg_file, zeroed_corr_file]
    ):
        arr = io.load_gdal(in_f)
        arr[mask == 0] = 0
        logger.debug(f"Size: {arr.size}, {(arr != 0).sum()} non-zero pixels")
        io.write_arr(
            arr=arr,
            output_name=out_f,
            like_filename=corr_filename,
        )
    return zeroed_ifg_file, zeroed_corr_file


def _redirect_unwrapping_log(unw_filename: Filename, method: str):
    import journal

    logfile = Path(unw_filename).with_suffix(".log")
    journal.info(f"isce3.unwrap.{method}").device = journal.logfile(
        fspath(logfile), "w"
    )
    logger.info(f"Logging unwrapping output to {logfile}")
