from __future__ import annotations

import itertools
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Pattern, Sequence, Union

import h5py
from shapely import geometry, ops, wkt

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

from .config import OPERA_BURST_RE, OPERA_DATASET_NAME, OPERA_IDENTIFICATION

logger = get_log(__name__)

__all__ = ["group_by_burst", "setup_output_folder"]


def group_by_burst(
    file_list: Sequence[Filename],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
    minimum_slcs: int = 2,
) -> dict[str, list[Path]]:
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: list[Filename]
        list of paths of CSLC files
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

    if not file_list:
        return {}

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
    strides: dict[str, int] = {"y": 1, "x": 1},
    creation_options: Optional[list] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> list[Path]:
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
    strides : dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    creation_options : list, optional
        list of options to pass to the GDAL driver, by default None
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    list[Path]
        list of saved empty files.
    """
    if output_folder is None:
        output_folder = vrt_stack.outfile.parent

    date_strs: list[str] = []
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


def get_cslc_polygon(
    opera_file: Filename, buffer_degrees: float = 0.0
) -> Union[geometry.Polygon, None]:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    dset_name = f"{OPERA_IDENTIFICATION}/bounding_polygon"
    with h5py.File(opera_file) as hf:
        if dset_name not in hf:
            logger.debug(f"Could not find {dset_name} in {opera_file}")
            return None
        wkt_str = hf[dset_name][()].decode("utf-8")
    return wkt.loads(wkt_str).buffer(buffer_degrees)


def get_union_polygon(
    opera_file_list: Sequence[Filename], buffer_degrees: float = 0.0
) -> geometry.Polygon:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    polygons = [get_cslc_polygon(f, buffer_degrees) for f in opera_file_list]
    polygons = [p for p in polygons if p is not None]

    if len(polygons) == 0:
        raise ValueError("No polygons found in the given file list.")
    # Union all the polygons
    return ops.union_all(polygons)


def make_nodata_mask(
    opera_file_list: Sequence[Filename],
    out_file: Filename,
    buffer_pixels: int = 0,
    overwrite: bool = False,
):
    """Make a dummy raster from the first file in the list.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    out_file : Filename
        Output filename.
    buffer_pixels : int, optional
        Number of pixels to buffer the union polygon by, by default 0.
        Note that buffering will *decrease* the numba of pixels marked as nodata.
        This is to be more conservative to not mask possible valid pixels.
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False
    """
    if Path(out_file).exists():
        if not overwrite:
            logger.debug(f"Skipping {out_file} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {out_file} since overwrite=True.")
            Path(out_file).unlink()

    # Check these are the right format to get nodata polygons
    try:
        test_f = f"NETCDF:{opera_file_list[0]}:{OPERA_DATASET_NAME}"
        io.get_raster_xysize(test_f)
    except RuntimeError:
        raise ValueError(f"Unable to open {test_f}")

    # convert pixels to degrees lat/lon
    # TODO: more robust way to get the pixel size... this is a hack
    # maybe just use pyproj to warp lat/lon to meters and back?
    if buffer_pixels == 0:
        buffer_degrees = 0.0
    else:
        gt = io.get_raster_gt(opera_file_list[0])
        dx_meters = gt[1]
        dx_degrees = dx_meters / 111000
        buffer_degrees = buffer_pixels * dx_degrees

    # Get the union of all the polygons and convert to a temp geojson
    union_poly = get_union_polygon(opera_file_list, buffer_degrees=buffer_degrees)
    # convert shapely polygon to geojson

    # Make a dummy raster from the first file with all 0s
    # This will get filled in with the polygon rasterization
    cmd = (
        f"gdal_calc.py --quiet --outfile {out_file} --type Byte  -A"
        f" NETCDF:{opera_file_list[0]}:{OPERA_DATASET_NAME} --calc 'numpy.nan_to_num(A)"
        " * 0' --creation-option COMPRESS=LZW --creation-option TILED=YES"
        " --creation-option BLOCKXSIZE=256 --creation-option BLOCKYSIZE=256"
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "geometry": geometry.mapping(union_poly),
                        "properties": {"id": 1},
                    }
                )
            )

        # Now burn in the union of all polygons
        cmd = f"gdal_rasterize -q -burn 1 {temp_vector_file} {out_file}"
        logger.info(cmd)
        subprocess.check_call(cmd, shell=True)
