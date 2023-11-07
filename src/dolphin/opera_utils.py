from __future__ import annotations

import itertools
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Pattern, Sequence, Union, overload

import h5py
import numpy as np
from shapely import geometry, ops, wkt

from dolphin._log import get_log
from dolphin._types import Filename, PathLikeT

logger = get_log(__name__)


# Specific to OPERA CSLC products:
OPERA_DATASET_ROOT = "/"
OPERA_DATASET_NAME = f"{OPERA_DATASET_ROOT}/data/VV"
OPERA_IDENTIFICATION = f"{OPERA_DATASET_ROOT}/identification"

# It should match either or these within a filename:
# t087_185684_iw2 (which comes from COMPASS)
# T087-165495-IW3 (which is the official product naming scheme)
# e.g.
# OPERA_L2_CSLC-S1_T078-165495-IW3_20190906T232711Z_20230101T100506Z_S1A_VV_v1.0.h5

OPERA_BURST_RE = re.compile(
    r"[tT](?P<track>\d{3})[-_](?P<burst_id>\d{6})[-_](?P<subswath>iw[1-3])",
    re.IGNORECASE,
)


def get_burst_id(
    filename: Filename, burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE
) -> str:
    """Extract the burst id from a filename.

    Matches either format of
        t087_185684_iw2 (which comes from COMPASS)
        T087-165495-IW3 (which is the official product naming scheme)

    Parameters
    ----------
    filename: Filename
        CSLC filename
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][dolphin.opera_utils.OPERA_BURST_RE]

    Returns
    -------
    str
        burst id of the SLC acquisition, normalized to be in the format
            t087_185684_iw2
    """
    if not (m := re.search(burst_id_fmt, str(filename))):
        raise ValueError(f"Could not parse burst id from {filename}")
    burst_str = m.group()
    # Normalize
    return burst_str.lower().replace("-", "_")


@overload
def group_by_burst(
    file_list: Iterable[str],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
) -> dict[str, list[str]]:
    ...


@overload
def group_by_burst(
    file_list: Iterable[PathLikeT],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
) -> dict[str, list[PathLikeT]]:
    ...


def group_by_burst(file_list, burst_id_fmt=OPERA_BURST_RE):
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: Iterable[Filename]
        list of paths of CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][dolphin.opera_utils.OPERA_BURST_RE]

    Returns
    -------
    dict
        key is the burst id of the SLC acquisition
        Value is a list of inputs which correspond to that burst:
        {
            't087_185678_iw2': ['inputs/t087_185678_iw2_20200101.h5',...,],
            't087_185678_iw3': ['inputs/t087_185678_iw3_20200101.h5',...,],
        }
    """
    if not file_list:
        return {}

    sorted_file_list = _sort_by_burst_id(list(file_list), burst_id_fmt)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    return grouped_images


@overload
def _sort_by_burst_id(file_list: Iterable[str], burst_id_fmt) -> list[str]:
    ...


@overload
def _sort_by_burst_id(file_list: Iterable[Path], burst_id_fmt) -> list[Path]:
    ...


def _sort_by_burst_id(file_list, burst_id_fmt):
    """Sort files/paths by burst id."""
    file_burst_tuples = sorted(
        [(f, get_burst_id(f, burst_id_fmt)) for f in file_list],
        # use the date or dates as the key
        key=lambda f_b_tuple: f_b_tuple[1],  # type: ignore
    )
    # Unpack the sorted pairs with new sorted values
    out_file_list = [f for f, _ in file_burst_tuples]
    return out_file_list


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
    return ops.unary_union(polygons)


def make_nodata_mask(
    opera_file_list: Sequence[Filename],
    out_file: Filename,
    buffer_pixels: int = 0,
    overwrite: bool = False,
):
    """Make a boolean raster mask from the union of nodata polygons.

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
    from dolphin import io

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
        # convert pixels to degrees lat/lon
        gt = io.get_raster_gt(test_f)
        # TODO: more robust way to get the pixel size... this is a hack
        # maybe just use pyproj to warp lat/lon to meters and back?
        dx_meters = gt[1]
        dx_degrees = dx_meters / 111000
        buffer_degrees = buffer_pixels * dx_degrees
    except RuntimeError:
        raise ValueError(f"Unable to open {test_f}")

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


@dataclass(frozen=True)
class BurstSubsetOption:
    """Dataclass for a possible subset of SLC data."""

    num_dates: int
    """Number of dates used in this subset"""
    num_burst_ids: int
    """Number of burst IDs used in this subset."""
    total_num_bursts: int
    """Total number of bursts used in this subset."""
    burst_id_list: list[str]
    """List of burst IDs used in this subset."""


def get_missing_data_options(
    slc_files: Optional[Iterable[Filename]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[str, date]]] = None,
) -> list[BurstSubsetOption]:
    """Get a list of possible data subsets for a set of burst SLCs.

    The default optimization criteria for choosing among these subsets is

        maximize        total number of bursts used
        subject to      dates used for each burst ID are all equal

    The constraint that the same dates are used for each burst ID is to
    avoid spatial discontinuities the estimated displacement/velocity,
    which can occur if different dates are used for different burst IDs.

    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        list of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, date]]]
        Alternative input: list of all existing (burst_id, date) tuples.

    Returns
    -------
    list[BurstSubsetOption]
        List of possible subsets of the given SLC data.
        The options will be sorted by the total number of bursts used, so
        that the first option is the one that uses the most data.
    """
    if slc_files is not None:
        burst_id_to_dates = _burst_id_mapping_from_files(slc_files)
    elif burst_id_date_tuples is not None:
        burst_id_to_dates = _burst_id_mapping_from_tuples(burst_id_date_tuples)
    else:
        raise ValueError("Must provide either slc_files or burst_id_date_tuples")

    all_dates = sorted(set(itertools.chain.from_iterable(burst_id_to_dates.values())))
    all_burst_ids = list(burst_id_to_dates.keys())

    # Construct the incidence matrix of dates vs. burst IDs
    burst_id_to_date_incidence = {}
    for burst_id, date_list in burst_id_to_dates.items():
        cur_incidences = np.zeros(len(all_dates), dtype=bool)
        idxs = np.searchsorted(all_dates, date_list)
        cur_incidences[idxs] = True
        burst_id_to_date_incidence[burst_id] = cur_incidences

    B = np.array(list(burst_id_to_date_incidence.values()))
    # In this matrix,
    # - Each column corresponds to one of the possible dates
    # - Each row corresponds to one of the possible burst IDs
    unique_date_idxs, burst_id_counts = np.unique(B, axis=0, return_counts=True)
    out = []

    for date_idxs in unique_date_idxs:
        required_num_dates = date_idxs.sum()
        keep_burst_idxs = np.array(
            [required_num_dates == burst_row[date_idxs].sum() for burst_row in B]
        )
        # B.shape: (num_burst_ids, num_dates)
        cur_burst_total = B[keep_burst_idxs, :][:, date_idxs].sum()

        cur_burst_id_list = np.array(all_burst_ids)[keep_burst_idxs].tolist()
        out.append(
            BurstSubsetOption(
                num_dates=required_num_dates,
                num_burst_ids=len(cur_burst_id_list),
                total_num_bursts=cur_burst_total,
                burst_id_list=cur_burst_id_list,
            )
        )
    return sorted(out, key=lambda x: x.total_num_bursts, reverse=True)


def _burst_id_mapping_from_tuples(
    burst_id_date_tuples: Iterable[tuple[str, date]]
) -> dict[str, list[date]]:
    """Create a {burst_id -> [date,...]} (burst_id, date) tuples."""
    # Don't exhaust the iterator for multiple groupings
    burst_id_date_tuples = list(burst_id_date_tuples)

    # Group the possible SLC files by their date and by their Burst ID
    return {
        burst_id: [date for burst_id, date in g]
        for burst_id, g in itertools.groupby(burst_id_date_tuples, key=lambda x: x[0])
    }


def _burst_id_mapping_from_files(
    slc_files: Iterable[Filename],
) -> dict[str, list[date]]:
    """Create a {burst_id -> [date,...]} mapping from filenames."""
    from dolphin.utils import get_dates

    # Don't exhaust the iterator for multiple groupings
    slc_file_list = list(map(str, slc_files))

    # Group the possible SLC files by their date and by their Burst ID
    burst_id_to_files = group_by_burst(slc_file_list)

    date_tuples = [get_dates(f) for f in slc_file_list]
    assert all(len(tup) == 1 for tup in date_tuples)

    return {
        burst_id: [get_dates(f)[0] for f in file_list]
        for (burst_id, file_list) in burst_id_to_files.items()
    }
