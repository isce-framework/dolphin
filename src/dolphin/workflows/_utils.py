import itertools
import re
import subprocess
from os import fspath
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Union

import h5py
import numpy as np
from osgeo import gdal
from shapely import wkt
from shapely.ops import unary_union

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

from .config import OPERA_BURST_RE, OPERA_DATASET_NAME

logger = get_log(__name__)

__all__ = ["group_by_burst", "setup_output_folder"]


def group_by_burst(
    file_list: Sequence[Filename],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
    minimum_slcs: int = 2,
) -> Dict[str, List[Path]]:
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: List[Filename]
        List of paths of CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][dolphin.workflows.config.OPERA_BURST_RE]
    minimum_slcs: int
        Minimum number of SLCs needed to run the workflow for each burst.
        If there are fewer SLCs in a burst, it will be skipped and
        a warning will be logged.

    Returns
    -------
    dict
        key is the burst id of the SLC acquisition
        Value is a list of Paths on that burst:
        {
            't087_185678_iw2': [Path(...), Path(...),],
            't087_185678_iw3': [Path(...),... ],
        }
    """

    def get_burst_id(filename):
        m = re.search(burst_id_fmt, str(filename))
        if not m:
            raise ValueError(f"Could not parse burst id from {filename}")
        return m.group()

    def sort_by_burst_id(file_list):
        """Sort files by burst id."""
        burst_ids = [get_burst_id(f) for f in file_list]
        file_burst_tups = sorted(
            [(Path(f), d) for f, d in zip(file_list, burst_ids)],
            # use the date or dates as the key
            key=lambda f_d_tuple: f_d_tuple[1],  # type: ignore
        )
        # Unpack the sorted pairs with new sorted values
        file_list, burst_ids = zip(*file_burst_tups)  # type: ignore
        return file_list

    sorted_file_list = sort_by_burst_id(file_list)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    # Make sure that each burst has at least the minimum number of SLCs
    out = {}
    for burst_id, slc_list in grouped_images.items():
        if len(slc_list) < minimum_slcs:
            logger.warning(
                f"Skipping burst {burst_id} because it has only {len(slc_list)} SLCs."
                f"Minimum number of SLCs is {minimum_slcs}"
            )
        else:
            out[burst_id] = slc_list
    return out


def setup_output_folder(
    vrt_stack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    strides: Dict[str, int] = {"y": 1, "x": 1},
    creation_options: Optional[List] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
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
    strides : Dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    creation_options : list, optional
        List of options to pass to the GDAL driver, by default None
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    List[Path]
        List of saved empty files.
    """
    if output_folder is None:
        output_folder = vrt_stack.outfile.parent

    date_strs: List[str] = []
    for d in vrt_stack.dates[start_idx:]:
        if len(d) == 1:
            # normal SLC files will have a single date
            s = d[0].strftime(io.DEFAULT_DATETIME_FORMAT)
        else:
            # Compressed SLCs will have 2 dates in the name marking the start and end
            s = io._format_date_pair(d[0], d[1])
        date_strs.append(s)

    output_files = []
    for filename in date_strs:
        slc_name = Path(filename).stem
        output_path = output_folder / f"{slc_name}.slc.tif"

        io.write_arr(
            arr=None,
            like_filename=vrt_stack.outfile,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
            options=creation_options,
        )

        output_files.append(output_path)
    return output_files


def get_union_polygon(opera_file_list: List[Filename], buffer_degrees: float = 0.0):
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file_list : List[Filename]
        List of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    polygons = []
    dset_name = "science/SENTINEL1/identification/bounding_polygon"
    for f in opera_file_list:
        with h5py.File(f) as hf:
            wkt_str = hf[dset_name][()].decode("utf-8")
        # geom = ogr.CreateGeometryFromWkt(wkt_str)
        geom = wkt.loads(wkt_str).buffer(buffer_degrees)
        polygons.append(geom)

    # Union all the polygons
    return unary_union(polygons)

    # total_poly = polygons[0]
    # for geom in polygons[1:]:
    #     total_poly = total_poly.Union(geom)
    # return total_poly


def intersection_over_union(poly1, poly2):
    """Compute the intersection over union of two polygons.

    Parameters
    ----------
    poly1 : shapely.geometry.Polygon
        First polygon.
    poly2 : shapely.geometry.Polygon
        Second polygon.

    Returns
    -------
    float
        Intersection over union of the two polygons.
    """
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)
    return intersection.area / union.area


def make_nodata_mask(
    opera_file_list: List[Filename], out_file: Filename, buffer_pixels: int = 0
):
    """Make a dummy raster from the first file in the list.

    Parameters
    ----------
    opera_file_list : List[Filename]
        List of COMPASS SLC filenames.
    out_file : Filename
        Output filename.
    buffer_pixels : int, optional
        Number of pixels to buffer the union polygon by, by default 0.
        Note that buffering will *decresase* the numba of pixels marked as nodata.
        This is to be more conservative to not mask possible valid pixels.
    """
    # buffer_pixels = min(buffer_pixels, min(ds.RasterXSize, ds.RasterYSize))
    # convert pixels to degrees lat/lon
    gt = io.get_raster_gt(opera_file_list[0])
    dx_meters = gt[1]
    dx_degrees = dx_meters / 111000
    buffer_degrees = buffer_pixels * dx_degrees

    # Get the union of all the polygons and convert to a temp geojson
    union_poly = get_union_polygon(opera_file_list, buffer_degrees=buffer_degrees)
    temp_vector = "temp.geojson"
    with open(temp_vector, "w") as f:
        f.write(union_poly.ExportToJson())

    # Make a dummy raster from the first file
    cmd = (
        f"gdal_calc.py --quiet --outfile {out_file} --type Byte  -A"
        f" NETCDF:{opera_file_list[0]}:{OPERA_DATASET_NAME} --calc 'numpy.nan_to_num(A)"
        " * 0'"
    )
    # TODO: Log and then run
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    # Now burn in the union of all polygons
    cmd = f"gdal_rasterize -burn 1 poly.geojson {out_file}"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    # Clean up the temp file
    Path(temp_vector).unlink()


def get_stack_nodata_mask(
    stack_filename: Filename,
    output_file: Optional[Filename] = None,
    compute_bands: Optional[List[int]] = None,
    buffer_pixels: int = 100,
    nodata: float = np.nan,
):
    """Get a mask of pixels that are nodata in all bands of `slc_stack_vrt`.

    Parameters
    ----------
    stack_filename : Path or str
        File containing the SLC stack as separate bands.
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
        logger.debug(f"Computing mask for band {b}")
        bnd = ds.GetRasterBand(b)
        arr = bnd.ReadAsArray()
        if np.isnan(nodata):
            nodata_mask = np.isnan(arr)
        else:
            nodata_mask = arr == nodata

        # Expand the region with a convolution
        if buffer_pixels > 0:
            logger.debug(f"Padding mask with {buffer_pixels} pixels")
            out_mask &= _erode_nodata(nodata_mask, buffer_pixels)
        else:
            out_mask &= nodata_mask

    if output_file:
        io.write_arr(
            arr=out_mask,
            output_name=output_file,
            like_filename=stack_filename,
            nbands=1,
            dtype="Byte",
        )
    return out_mask


def _erode_nodata(nd_mask, buffer_pixels=25):
    """Erode the nodata mask by `buffer_pixels`.

    This makes the nodata mask more conservative:
    there will be fewer pixels marked as nodata after.

    Parameters
    ----------
    nd_mask : np.ndarray[bool]
        Array where True indicates nodata.
    buffer_pixels : int, optional
        Size (in pixels) of erosion structural element to use.
        By default 25.

    Returns
    -------
    np.ndarray[bool]
        Same size as `nd_mask`, with no data pixels shrunk
        after erosion.
    """
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
