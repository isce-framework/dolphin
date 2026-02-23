from __future__ import annotations

import logging
import tempfile
import warnings
from enum import IntEnum
from os import fspath
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer
from shapely import from_wkt, geometry, ops, to_geojson

from dolphin import io
from dolphin._types import Bbox, PathOrStr

gdal.UseExceptions()

logger = logging.getLogger("dolphin")


class MaskingError(ValueError):
    """Exception indicating the mask was improperly built, no valid pixels remain."""


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


def combine_mask_files(
    mask_files: Sequence[PathOrStr],
    output_file: PathOrStr,
    dtype: str = "uint8",
    output_convention: MaskConvention = MaskConvention.ZERO_IS_NODATA,
    input_conventions: Sequence[MaskConvention] | None = None,
    combine_method: Literal["any", "all"] = "any",
    raise_on_empty: bool = True,
):
    """Combine multiple mask files into a single mask file.

    All `mask_files` must be the same size and projected on the same grid.

    Parameters
    ----------
    mask_files : list of Path or str
        list of mask files to combine.
    output_file : PathOrStr
        Path to the combined output file.
    dtype : str, optional
        Data type of output file. Default is uint8.
    output_convention : MaskConvention, optional
        Convention to use for output mask. Default is SNAPHU,
        where 0 indicates invalid pixels.
    input_conventions : list of MaskConvention, optional
        Convention to use for each input mask. Default is None,
        where it is assumed all masks use the "0 is invalid" convention.
    combine_method : str, optional, default = 'any', choices = ['any', 'all']
        Logical operation to use to combine masks. Default is 'any',
        which means the output is masked where *any* of the input masks indicated
        a masked pixel (the masked region grows larger).
        If 'all', the only pixels masked are those in which *all* input masks
        indicated a masked pixel (the masked region shrinks).
    raise_on_empty : bool
        If True, raises a `MaskingError` on the creation of a mask file with
        no valid pixels.
        Otherwise, raises a warning.
        Default is True.

    Raises
    ------
    ValueError
        If `input_conventions` passed and is different length as `mask_files`
        If all mask_files are not the same shape

    """
    output_file = Path(output_file)
    xsize, ysize = io.get_raster_xysize(mask_files[0])

    if combine_method not in ["any", "all"]:
        msg = "combine_method must be 'any' or 'all'"
        raise ValueError(msg)

    if input_conventions is None:
        input_conventions = [MaskConvention.ZERO_IS_NODATA] * len(mask_files)
    elif isinstance(input_conventions, MaskConvention):
        input_conventions = [input_conventions] * len(mask_files)

    if len(input_conventions) != len(mask_files):
        msg = (
            f"input_conventions ({len(input_conventions)}) must have the same"
            f" length as mask_files ({len(mask_files)})"
        )
        raise ValueError(msg)

    from dolphin.io._blocks import iter_blocks

    block_shape = (512, 512)

    # Pre-create the output file
    io.write_arr(
        arr=None,
        output_name=output_file,
        like_filename=mask_files[0],
        dtype=dtype,
        nbands=1,
        nodata=None,
    )

    num_valid = 0
    for block in iter_blocks(
        arr_shape=(ysize, xsize),
        block_shape=block_shape,
    ):
        rows = slice(block.row_start, block.row_stop)
        cols = slice(block.col_start, block.col_stop)
        block_rows = block.row_stop - block.row_start
        block_cols = block.col_stop - block.col_start

        # Uses the numpy convention (1 = invalid, 0 = valid) for combining logic
        if combine_method == "any":
            mask_total = np.zeros((block_rows, block_cols), dtype=bool)
        else:
            mask_total = np.ones((block_rows, block_cols), dtype=bool)

        for input_convention, mask_file in zip(
            input_conventions, mask_files, strict=False
        ):
            mask = io.load_gdal(mask_file, rows=rows, cols=cols)
            nd = io.get_raster_nodata(mask_file)
            if nd is not None:
                nodata_pixels = (
                    np.isnan(mask) if np.isnan(nd) else (mask == nd)
                )
            else:
                nodata_pixels = np.zeros_like(mask, dtype=bool)
            mask = mask.astype(bool)
            # Fill nodata with the convention's "mask" value
            mask[nodata_pixels] = bool(input_convention.value)
            if input_convention != MaskConvention.NUMPY:
                mask = ~mask

            if combine_method == "any":
                np.logical_or(mask_total, mask, out=mask_total)
            elif combine_method == "all":
                np.logical_and(mask_total, mask, out=mask_total)

        num_valid += mask_total.size - mask_total.sum()

        # Convert to output convention
        if output_convention == MaskConvention.SNAPHU:
            mask_total = ~mask_total

        io.write_block(
            mask_total.astype(dtype), output_file, block.row_start, block.col_start
        )

    if num_valid == 0:
        msg = "No valid pixels left in mask"
        if raise_on_empty:
            raise MaskingError(msg)
        else:
            warnings.warn(msg, stacklevel=2)


def load_mask_as_numpy(mask_file: PathOrStr) -> _LazyMask:
    """Load `mask_file` as a lazy block-reading boolean array.

    The returned object supports ``arr[row_slice, col_slice]`` and ``.all()``
    so it can be used anywhere the previous full-array return was used, while
    avoiding loading the entire raster into memory at once.

    The mask file is expected to use 0 for invalid data and 1 for good data.
    The output follows the numpy masking convention where True indicates
    nodata and False indicates valid data (i.e. the values are inverted).

    Parameters
    ----------
    mask_file : PathOrStr
        Path to the mask file.

    Returns
    -------
    _LazyMask
        A lazy array-like where True values indicate nodata (invalid) pixels
        and False values indicate valid data pixels.

    """
    return _LazyMask(mask_file)


class _LazyMask:
    """Lazy raster reader for boolean masks.

    Reads blocks on demand via ``__getitem__``, avoiding full-array allocation.
    Supports the ``[row_slice, col_slice]`` and ``.all()`` interface that
    callers of :func:`load_mask_as_numpy` rely on.
    """

    def __init__(self, filename: PathOrStr):
        self.filename = filename
        self._nodata = io.get_raster_nodata(filename)
        cols, rows = io.get_raster_xysize(filename)
        self.shape = (rows, cols)

    def __getitem__(self, key):
        rows, cols = key
        block = io.load_gdal(self.filename, rows=rows, cols=cols)
        nd = self._nodata
        if nd is not None:
            if np.isnan(nd):
                nodata_pixels = np.isnan(block)
            else:
                nodata_pixels = block == nd
        else:
            nodata_pixels = np.zeros_like(block, dtype=bool)
        # Convention: 0=invalid → True (nodata), non-zero=valid → False
        result = ~block.astype(bool)
        # Nodata pixels should also be True (masked)
        result[nodata_pixels] = True
        return result

    def all(self):
        """Check if all pixels are masked (nodata)."""
        from dolphin.io._blocks import iter_blocks

        for block in iter_blocks(arr_shape=self.shape, block_shape=(512, 512)):
            chunk = self[
                slice(block.row_start, block.row_stop),
                slice(block.col_start, block.col_stop),
            ]
            if not chunk.all():
                return False
        return True


def create_bounds_mask(
    bounds: Bbox | tuple[float, float, float, float] | None,
    bounds_wkt: str | None,
    output_filename: PathOrStr,
    like_filename: PathOrStr,
    bounds_epsg: int = 4326,
    overwrite: bool = False,
) -> None:
    """Create a boolean raster mask where 1 is inside the given bounds and 0 is outside.

    Parameters
    ----------
    bounds : tuple, optional
        (min x, min y, max x, max y) of the area of interest
    bounds_wkt : tuple, optional
        Well known text (WKT) string describing Polygon of the area of interest.
        Alternative to bounds. Cannot pass both bounds and bounds_wkt.
    like_filename : Filename
        Reference file to copy the shape, extent, and projection.
    output_filename : Filename
        Output filename for the mask
    bounds_epsg : int, optional
        EPSG code of the coordinate system of the bounds.
        Default is 4326 (lat/lon coordinates for the bounds).
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False

    Raises
    ------
    ValueError
        If neither bounds nor bounds_wkt, or both bounds and bounds_wkt, is passed.

    """
    if bounds is None and bounds_wkt is None:
        raise ValueError("Must pass either `bounds` or `bounds_wkt`")
    if bounds is not None and bounds_wkt is not None:
        raise ValueError("Cannot pass both `bounds` and `bounds_wkt`")

    if Path(output_filename).exists():
        if not overwrite:
            logger.info(f"Skipping {output_filename} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {output_filename} since overwrite=True.")
            Path(output_filename).unlink()

    # Create a polygon from the bounds or wkt
    bounds_poly = geometry.box(*bounds) if bounds is not None else from_wkt(bounds_wkt)

    # Geojson default is 4326, so we need it in that system
    # Transform bounds if necessary
    if bounds_epsg != 4326:
        # Pass the Pyproj transformer to `shapely.ops.transform`
        transformer = Transformer.from_crs(
            CRS.from_epsg(bounds_epsg), 4326, always_xy=True
        )
        bounds_poly_lonlat = ops.transform(transformer.transform, bounds_poly)
    else:
        bounds_poly_lonlat = bounds_poly

    logger.info(f"Creating mask for bounds {bounds_poly_lonlat}")

    # Create the output raster
    io.write_arr(
        arr=None,
        output_name=output_filename,
        dtype=bool,
        nbands=1,
        like_filename=like_filename,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w") as f:
            f.write(to_geojson(bounds_poly_lonlat))

        # Open the input vector file
        src_ds = gdal.OpenEx(fspath(temp_vector_file), gdal.OF_VECTOR)
        dst_ds = gdal.Open(fspath(output_filename), gdal.GA_Update)

        # Now burn in the union of all polygons
        gdal.Rasterize(dst_ds, src_ds, burnValues=[1])
        src_ds = dst_ds = None

    logger.info(f"Created {output_filename}")
