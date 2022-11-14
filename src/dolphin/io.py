from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal

from dolphin.log import get_log
from dolphin.utils import Pathlike, numpy_to_gdal_type
from dolphin.vrt import VRTStack

gdal.UseExceptions()
logger = get_log()


DEFAULT_TILE_SIZE = [128, 128]
DEFAULT_TIFF_OPTIONS = [
    "COMPRESS=DEFLATE",
    "ZLEVEL=5",
    "TILED=YES",
    f"BLOCKXSIZE={DEFAULT_TILE_SIZE[1]}",
    f"BLOCKYSIZE={DEFAULT_TILE_SIZE[0]}",
]


def load_gdal(filename, band=None):
    """Load a gdal file into a numpy array."""
    ds = gdal.Open(fspath(filename))
    return ds.ReadAsArray() if band is None else ds.GetRasterBand(band).ReadAsArray()


def copy_projection(src_file: Pathlike, dst_file: Pathlike) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    ds_src = gdal.Open(fspath(src_file))
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        logger.info("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def save_arr_like(*, arr, like_filename, output_name, driver="GTiff", options=None):
    """Save an array to a file, copying projection/nodata from `like_filename`."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    ysize, xsize = arr.shape[-2:]
    nbands = arr.shape[0]

    ds = gdal.Open(fspath(like_filename))
    if driver is None:
        driver = ds.GetDriver().ShortName
    if options is None and driver == "GTiff":
        options = DEFAULT_TIFF_OPTIONS
    drv = gdal.GetDriverByName(driver)
    out_ds = drv.Create(
        fspath(output_name),
        xsize,
        ysize,
        nbands,
        numpy_to_gdal_type(arr.dtype),
        options=options or [],
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    for i in range(nbands):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[i])
    # TODO: copy other metadata
    ds = out_ds = None


def setup_output_folder(
    vrt_stack: VRTStack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    make_compressed: bool = True,
    creation_options: Optional[List] = None,
) -> Tuple[List[Path], Optional[Path]]:
    """Create empty output files for each band in `vrt_stack`.

    Also creates an empty file for the compressed SLC.
    Used to prepare output for block processing.

    Parameters
    ----------
    vrt_stack : VRTStack
        object containing the current stack of SLCs
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    start_idx : int, optional
        Index of vrt_stack to begin making output files.
        This should match the ministack index to avoid re-creating the
        past compressed SLCs.
    make_compressed : bool, optional
        Whether to create an empty file for the compressed SLC, by default True
    creation_options : list, optional
        List of options to pass to the GDAL driver, by default None

    Returns
    -------
    List[Path]
        List of saved empty files.
    """
    output_folder = vrt_stack.outfile.parent
    _, ysize, xsize = vrt_stack.shape

    if creation_options is None and driver == "GTiff":
        creation_options = DEFAULT_TIFF_OPTIONS

    in_ds = gdal.Open(fspath(vrt_stack.file_list[start_idx]))
    output_files = []
    for filename in vrt_stack.file_list[start_idx:]:
        slc_name = Path(filename).stem
        # TODO: get extension from cfg
        output_path = output_folder / f"{slc_name}.slc.tif"

        drv = gdal.GetDriverByName(driver)
        out_ds = drv.Create(
            fspath(output_path),
            xsize,
            ysize,
            1,
            numpy_to_gdal_type(dtype),
            options=creation_options,
        )
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        out_ds.SetProjection(in_ds.GetProjection())
        # TODO: copy other metadata

        output_files.append(output_path)
        out_ds.FlushCache()
        out_ds = None

    if not make_compressed:
        # Skip creating the compressed SLC
        in_ds = None
        return output_files, None

    # Make the compressed SLC file
    # Note: this is the first compressed SLC in the next ministack,
    # and it will be in correct sorted order assuming `start_end`
    # is before the next ministack's start
    start_end = output_folder.name
    comp_slc_file = output_folder / f"compressed_{start_end}.tif"

    drv = gdal.GetDriverByName(driver)
    out_ds = drv.Create(
        fspath(comp_slc_file),
        xsize,
        ysize,
        1,
        numpy_to_gdal_type(dtype),
        options=creation_options or [],
    )
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.FlushCache()
    # TODO: copy other metadata

    in_ds = out_ds = None
    return output_files, comp_slc_file


def save_block(
    cur_block: np.ndarray,
    output_files: Union[Path, List[Path]],
    rows: slice,
    cols: slice,
):
    """Save each of the MLE estimates (ignoring the compressed SLCs).

    Parameters
    ----------
    cur_block : np.ndarray
        Array of shape (n_bands, block_rows, block_cols)
    output_files : List[Path] or Path
        List of output files to save to, or (if cur_block is 2D) a single file.
    rows : slice
        Rows of the current block
    cols : slice
        Columns of the current block

    Raises
    ------
    ValueError
        If length of `output_files` does not match length of `cur_block`.
    """
    if not isinstance(output_files, list):
        output_files = [output_files]
    if cur_block.ndim == 2:
        # Make into 3D array shaped (1, rows, cols)
        cur_block = cur_block[np.newaxis, ...]

    if len(cur_block) != len(output_files):
        raise ValueError(
            f"cur_block has {len(cur_block)} layers, but passed"
            f" {len(output_files)} files"
        )
    for cur_image, filename in zip(cur_block, output_files):
        ds = gdal.Open(fspath(filename), gdal.GA_Update)
        bnd = ds.GetRasterBand(1)
        # only need offset for write:
        # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Band.WriteArray
        bnd.WriteArray(cur_image, cols.start, rows.start)
        bnd.FlushCache()
        bnd = ds = None
