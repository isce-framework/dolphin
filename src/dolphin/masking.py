from enum import IntEnum
from os import fspath
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from osgeo import gdal

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.utils import numpy_to_gdal_type

gdal.UseExceptions()

logger = get_log(__name__)


class MaskConvention(IntEnum):
    """Enum for masking conventions to indicate the nodata value as 0 or 1.

    In numpy, 1 indicates nodata
    https://numpy.org/doc/stable/reference/maskedarray.generic.html#what-is-a-masked-array

    In SNAPHU (and isce2, and some water masks), the binary mask you pass has the
    opposite convention, where 0s mean the pixel is invalid (nodata).
    """

    ZERO_IS_NODATA = 0
    ONE_IS_NODATA = 1

    # Semantic aliases for the above:
    SNAPHU = 0
    NUMPY = 1

    def get_nodata_value(self) -> int:
        """Return the nodata value for this convention."""
        return int(not self.value)


def warp_to_match(
    input_file: Filename,
    match_file: Filename,
    output_file: Optional[Filename] = None,
    resampling_alg: str = "near",
    output_format: Optional[str] = None,
) -> Path:
    """Reproject a mask image to align with the input image.

    Parameters
    ----------
    input_file: Filename
        Path to the mask image to be reprojected.
    match_file: Filename
        Path to the input image to serve as a reference for the reprojected mask.
        Uses the bounds, resolution, and CRS of this image.
    output_file: Filename
        Path to the output image which is the reprojected water mask image.
        If None, creates an in-memory warped VRT.
    resampling_alg: str, optional, default = "near"
        Resampling algorithm to be used during reprojection.
        See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for choices.
    output_format: str, optional, default = None
        Output format to be used for the output image.
        If None, gdal will try to infer the format from the output file extension, or
        (if the extension of `output_file` matches `input_file`) use the input driver.

    Returns
    -------
    Path
        Path to the output image.
        Same as `output_file` if provided, otherwise a path to the in-memory VRT.
    """
    bounds = io.get_raster_bounds(match_file)
    crs_wkt = io.get_raster_crs(match_file).to_wkt()
    gt = io.get_raster_gt(match_file)
    resolution = (gt[1], gt[5])

    if output_file is None:
        output_file = f"/vsimem/warped_{Path(input_file).stem}.vrt"
        logger.debug(f"Creating in-memory warped VRT: {output_file}")

    if output_format is None and Path(input_file).suffix == Path(output_file).suffix:
        output_format = io.get_raster_driver(input_file)

    options = gdal.WarpOptions(
        dstSRS=crs_wkt,
        format=output_format,
        xRes=resolution[0],
        yRes=resolution[1],
        outputBounds=bounds,
        outputBoundsSRS=crs_wkt,
        resampleAlg=resampling_alg,
    )
    gdal.Warp(
        fspath(output_file),
        fspath(input_file),
        options=options,
    )

    return Path(output_file)


def combine_mask_files(
    mask_files: Sequence[Filename],
    output_file: Filename,
    dtype: str = "uint8",
    output_convention: MaskConvention = MaskConvention.SNAPHU,
    input_conventions: Optional[Sequence[MaskConvention]] = None,
    combine_method: str = "any",
    output_format: str = "GTiff",
):
    """Combine multiple mask files into a single mask file.

    Parameters
    ----------
    mask_files : list of Path or str
        List of mask files to combine.
    output_file : Filename
        Path to the combined output file.
    dtype : str, optional
        Data type of output file. Default is uint8.
    output_convention : MaskConvention, optional
        Convention to use for output mask. Default is SNAPHU,
        where 0 indicates invalid pixels.
    input_conventions : list of MaskConvention, optional
        Convention to use for each input mask. Default is None,
        where it is assumed all masks use the numpy convention
        (1 indicates invalid pixels).
    combine_method : str, optional, default = 'any', choices = ['any', 'all']
        Logical operation to use to combine masks. Default is 'any',
        which means the output is masked where *any* of the input masks indicated
        a masked pixel (the masked region grows larger).
        If 'all', the only pixels masked are those in which *all* input masks
        indicated a masked pixel (the masked region shrinks).
    output_format : str, optional, default = 'GTiff'
        Output format to be used for the output image.

    Raises
    ------
    ValueError
        If `input_conventions` passed and is different length as `mask_files`
        If all mask_files are not the same shape
    """
    output_file = Path(output_file)
    gt = io.get_raster_gt(mask_files[0])
    crs = io.get_raster_crs(mask_files[0])
    xsize, ysize = io.get_raster_xysize(mask_files[0])

    if combine_method not in ["any", "all"]:
        raise ValueError("combine_method must be 'any' or 'all'")

    # Create output file
    driver = gdal.GetDriverByName(output_format)
    ds_out = driver.Create(
        fspath(output_file),
        xsize,
        ysize,
        1,
        numpy_to_gdal_type(dtype),
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(crs.to_wkt())
    # TODO: we probably want a separate "mask nodata", where there was
    # no data in the original, to be separate from the "bad data" value of 0/1,
    # similar to the PS masking we set up.
    # ds_out.GetRasterBand(1).SetNoDataValue(int(output_convention))

    if input_conventions is None:
        input_conventions = [MaskConvention.NUMPY] * len(mask_files)
    elif len(input_conventions) != len(mask_files):
        raise ValueError(
            (
                f"input_conventions ({len(input_conventions)}) must have the same"
                f" length as mask_files ({len(mask_files)})"
            ),
        )

    # Uses the numpy convention (1 = invalid, 0 = valid) for combining logic
    # Loop through mask files and update the total mask
    if combine_method == "any":
        # "any" will use `logical_or` to grow the region starting empty region (as 0s)
        mask_total = np.zeros((ysize, xsize), dtype=bool)
    elif combine_method == "all":
        # "and" uses `logical_and` to shrink the full starting region (as 1s)
        mask_total = np.ones((ysize, xsize), dtype=bool)

    for input_convention, mask_file in zip(input_conventions, mask_files):
        # TODO: if we separate input missing data from mask 1/0, this changes
        mask = io.load_gdal(mask_file, masked=True).astype(bool)
        # Fill with "mask" value
        mask = mask.filled(bool(input_convention.value))
        if input_convention != MaskConvention.NUMPY:
            mask = ~mask

        if combine_method == "any":
            mask_total = np.logical_or(mask_total, mask)
        elif combine_method == "all":
            mask_total = np.logical_and(mask_total, mask)

    # Convert to output convention
    if output_convention == MaskConvention.SNAPHU:
        mask_total = ~mask_total
    ds_out.GetRasterBand(1).WriteArray(mask_total.astype(dtype))
    ds_out = None
