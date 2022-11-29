from os import fspath
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from osgeo import gdal

from dolphin.log import get_log
from dolphin.utils import Pathlike, gdal_to_numpy_type, numpy_to_gdal_type

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


def load_gdal(
    filename: Pathlike, band: Optional[int] = None, subsample_factor: int = 1
):
    """Load a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int, optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.
    """
    ds = gdal.Open(fspath(filename))
    rows, cols = ds.RasterYSize, ds.RasterXSize
    # Make an output object of the right size
    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty(
            (count, rows // subsample_factor, cols // subsample_factor), dtype=dt
        )
        ds.ReadAsArray(buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((rows // subsample_factor, cols // subsample_factor), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(buf_obj=out, resample_alg=resamp)
    return out


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


def get_nodata(filename: Pathlike) -> Optional[float]:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : Pathlike
        Path to the file to load.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.
    """
    ds = gdal.Open(fspath(filename))
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    return nodata


def save_arr(
    *,
    arr: Optional[np.ndarray],
    output_name: Pathlike,
    like_filename: Optional[Pathlike] = None,
    driver: Optional[str] = "GTiff",
    options: Optional[List] = None,
    nbands: Optional[int] = None,
    dtype: Optional[Union[str, np.dtype, type]] = None,
):
    """Save an array to `output_name`.

    If `like_filename` if provided, copies the projection/nodata.
    Options can be overridden by passing `driver`/`nbands`/`dtype`.

    If arr is None, create an empty file with the same x/y shape as `like_filename`.

    Parameters
    ----------
    arr : np.ndarray, optional
        Array to save. If None, create an empty file.
    output_name : str or Path
        Path to save the file to.
    like_filename : str or Path, optional
        Path to a file to copy raster shape/metadata from.
    driver : str, optional
        GDAL driver to use. Default is "GTiff".
    options : list, optional
        List of options to pass to the driver. Default is DEFAULT_TIFF_OPTIONS.
    nbands : int, optional
        Number of bands to save. Default is 1.
    dtype : str or np.dtype or type, optional
        Data type to save. Default is `arr.dtype` or the datatype of like_filename.
    """
    if like_filename is not None:
        ds_like = gdal.Open(fspath(like_filename))
    else:
        ds_like = None

    if arr is not None:
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        ysize, xsize = arr.shape[-2:]
        gdal_dtype = numpy_to_gdal_type(arr.dtype)
    else:
        if not ds_like:
            raise ValueError("Must specify either `arr` or `like_filename`")
        xsize, ysize = ds_like.RasterXSize, ds_like.RasterYSize
        if dtype is not None:
            gdal_dtype = numpy_to_gdal_type(dtype)
        else:
            gdal_dtype = ds_like.GetRasterBand(1).DataType

    nbands = nbands or (ds_like.RasterCount if ds_like else arr.shape[0])

    if driver is None:
        if str(output_name).endswith(".tif"):
            driver = "GTiff"
        else:
            if not ds_like:
                raise ValueError("Must specify `driver` if `like_filename` is None")
            driver = ds_like.GetDriver().ShortName
    if options is None and driver == "GTiff":
        options = DEFAULT_TIFF_OPTIONS

    drv = gdal.GetDriverByName(driver)
    ds_out = drv.Create(
        fspath(output_name),
        xsize,
        ysize,
        nbands,
        gdal_dtype,
        options=options or [],
    )

    if ds_like:
        ds_out.SetGeoTransform(ds_like.GetGeoTransform())
        ds_out.SetProjection(ds_like.GetProjection())

    # Write the actual data
    if arr is not None:
        for i in range(nbands):
            print(f"Writing band {i+1}/{nbands}")
            ds_out.GetRasterBand(i + 1).WriteArray(arr[i])
    # TODO: copy other metadata
    ds_out.FlushCache()
    ds_like = ds_out = None


def setup_output_folder(
    vrt_stack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    creation_options: Optional[List] = None,
) -> List[Path]:
    """Create empty output files for each band after `start_idx` in `vrt_stack`.

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
    creation_options : list, optional
        List of options to pass to the GDAL driver, by default None

    Returns
    -------
    List[Path]
        List of saved empty files.
    """
    output_folder = vrt_stack.outfile.parent

    output_files = []
    date_strs = [d.strftime("%Y%m%d") for d in vrt_stack.dates]

    for filename in date_strs[start_idx:]:
        slc_name = Path(filename).stem
        # TODO: get extension from cfg
        # output_path = output_folder / f"{slc_name}.slc.tif"
        output_path = output_folder / f"{slc_name}.slc.tif"

        save_arr(
            arr=None,
            like_filename=vrt_stack.outfile,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            options=creation_options,
        )

        output_files.append(output_path)
    return output_files


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


def get_stack_nodata_mask(
    stack_filename: Pathlike,
    output_file: Optional[Pathlike] = None,
    compute_bands: Optional[List[int]] = None,
    buffer_pixels: int = 100,
    nodata: float = np.nan,
):
    """Get a mask of pixels that are nodata in all bands of `slc_stack_vrt`.

    Parameters
    ----------
    stack_filename : Path or str
        VRTStack object containing the SLC stack.
    output_file : Path or str, optional
        Name of file to save to., by default None
    compute_bands : List[int], optional
        List of bands in vrt_stack to read.
        If None, reads in the first, middle, and last images.
    buffer_pixels : int, optional
        Number of pixels to expand the good-data area, by default 100
    nodata : float, optional
        Value of no data in the vrt_stack, by default np.nan

    Returns
    -------
    mask : np.ndarray[bool]
        Array where True indicates all bands are nodata.
    """
    ds = gdal.Open(fspath(stack_filename))
    if compute_bands is None:
        count = ds.RasterCount
        # Get the first and last file
        compute_bands = list(sorted(set([1, count])))

    # Start with ones, then only keep pixels that are nodata
    # in all the bands we check (reducing using logical_and)
    out_mask = np.ones((ds.RasterYSize, ds.RasterXSize), dtype=bool)

    # cap buffer pixel length to be no more the image size
    buffer_pixels = min(buffer_pixels, min(ds.RasterXSize, ds.RasterYSize))
    for b in compute_bands:
        print(f"Computing mask for band {b}")
        bnd = ds.GetRasterBand(b)
        arr = bnd.ReadAsArray()
        if np.isnan(nodata):
            nodata_mask = np.isnan(arr)
        else:
            nodata_mask = arr == nodata

        # Expand the region with a convolution
        if buffer_pixels > 0:
            print(f"Padding mask with {buffer_pixels} pixels")
            out_mask &= _erode_nodata(nodata_mask, buffer_pixels)
        else:
            out_mask &= nodata_mask

    if output_file:
        save_arr(
            arr=out_mask,
            output_name=output_file,
            like_filename=stack_filename,
            nbands=1,
            dtype="Byte",
        )
    return out_mask


def _erode_nodata(nd_mask, buffer_pixels=25):
    # invert so that good pixels are 1
    # we want to expand the area that is considered "good"
    # since we're being conservative with what we completely ignore
    out = (~nd_mask).astype("float32").copy()
    strel = np.ones(buffer_pixels)
    for i in range(out.shape[0]):
        o = np.convolve(out[i, :], strel, mode="same")
        out[i, :] = o
    for j in range(out.shape[1]):
        o = np.convolve(out[:, j], strel, mode="same")
        out[:, j] = o
    # convert back to binary mask, and re-invert
    return ~(out > 1e-3)
