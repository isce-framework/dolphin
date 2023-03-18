"""Module for creating the OPERA output product in NetCDF format."""
from os import fspath
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyproj
from numpy.typing import ArrayLike
from osgeo import gdal

from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.io import DEFAULT_HDF5_OPTIONS

logger = get_log(__name__)


BASE_GROUP = "/science/SENTINEL1"
DISP_GROUP = f"{BASE_GROUP}/DISP"
CORRECTIONS_GROUP = f"{BASE_GROUP}/corrections"
GLOBAL_ATTRS = dict(
    Conventions="CF-1.8",
    contact="operaops@jpl.nasa.gov",
    institution="NASA JPL",
    mission_name="OPERA",
    reference_document="TBD",
    title="OPERA L3_DISP_S1 Product",
)
GRID_MAPPING_DSET = "spatial_ref"
# Convert chunks to a tuple or h5py errors
HDF5_OPTS = DEFAULT_HDF5_OPTIONS.copy()
HDF5_OPTS["chunks"] = tuple(HDF5_OPTS["chunks"])  # type: ignore


def create_output_product(
    unw_filename: Filename,
    conncomp_filename: Filename,
    output_name: Filename,
    corrections: Optional[Dict[str, ArrayLike]] = None,
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
    unw_filename : Filename
        The path to the input unwrapped phase image.
    conncomp_filename : Filename
        The path to the input connected components image.
    output_name : Filename, optional
        The path to the output NetCDF file, by default "output.nc"
    corrections : Dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    """
    # Read the Geotiff file and its metadata
    displacement_ds = gdal.Open(fspath(unw_filename))
    gt = displacement_ds.GetGeoTransform()
    crs = pyproj.CRS.from_wkt(displacement_ds.GetProjection())
    unw_arr = displacement_ds.ReadAsArray()
    displacement_ds = None

    # Get the nodata mask (which for snaphu is 0)
    mask = unw_arr == 0
    # Set to NaN for final output
    unw_arr[mask] = np.nan

    conncomp_ds = gdal.Open(fspath(conncomp_filename))
    conncomp_arr = conncomp_ds.ReadAsArray()
    conncomp_ds = None
    # the conncomp nodata is 0

    fill_values = {
        "unwrapped_phase": np.nan,
        "connected_components": 0,
    }

    with h5py.File(output_name, "w") as f:
        # Create the NetCDF file
        f.attrs.update(GLOBAL_ATTRS)

        # Create the '/science/SENTINEL1/DISP/grids/displacement' group
        displacement_group = f.create_group(DISP_GROUP)

        # Set up the grid mapping variable
        _create_grid_mapping(displacement_group, crs, gt)

        # Set up the X/Y variables
        x_ds, y_ds = _create_xy_dsets(displacement_group, gt, unw_arr.shape)

        # Write the displacement array / conncomp arrays
        for img, (name, fv) in zip([unw_arr, conncomp_arr], fill_values.items()):
            dset = displacement_group.create_dataset(
                name,
                data=img,
                fillvalue=fv,
                **HDF5_OPTS,
            )
            dset.attrs["grid_mapping"] = GRID_MAPPING_DSET
            # # Attach the X/Y coordinates
            dset.dims[0].attach_scale(y_ds)
            dset.dims[1].attach_scale(x_ds)

        # Create the '/science/SENTINEL1/DISP/corrections' group
        corrections_group = f.create_group(CORRECTIONS_GROUP)
        if corrections:
            # Write the tropospheric/ionospheric correction images (if they exist)
            _create_correction_dsets(corrections_group, corrections)


def _create_xy(
    gt: List[float], shape: Tuple[int, int]
) -> Tuple[h5py.Dataset, h5py.Dataset]:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)
    return x, y


def _create_xy_dsets(
    group: h5py.Group, gt: List[float], shape: Tuple[int, int]
) -> Tuple[h5py.Dataset, h5py.Dataset]:
    """Create the x and y coordinate datasets."""
    x, y = _create_xy(gt, shape)

    # Create the datasets
    x_ds = group.create_dataset("x_coordinates", data=x, dtype=float)
    y_ds = group.create_dataset("y_coordinates", data=y, dtype=float)

    for name, ds in zip(["x", "y"], [x_ds, y_ds]):
        # ds.make_scale(name)
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name} coordinate of projection"
        ds.attrs["units"] = "m"

    return x_ds, y_ds


def _create_grid_mapping(group, crs: pyproj.CRS, gt: List[float]) -> h5py.Dataset:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_dataset(GRID_MAPPING_DSET, data=0, dtype=int)

    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs["GeoTransform"] = gt_string
    return dset


def _create_correction_dsets(
    corrections_group: h5py.Group, corrections: Dict[str, ArrayLike]
):
    """Create datasets for the tropospheric/ionospheric/other corrections."""
    troposphere = corrections.get("troposphere")
    if troposphere:
        troposphere_dset = corrections_group.create_dataset(
            "troposphere", data=troposphere, **HDF5_OPTS
        )
        troposphere_dset.attrs["grid_mapping"] = "crs"

    ionosphere = corrections["ionosphere"]
    if ionosphere:
        # Write the ionosphere correction image
        ionosphere_dset = corrections_group.create_dataset(
            "ionosphere", data=ionosphere, **HDF5_OPTS
        )
        ionosphere_dset.attrs["grid_mapping"] = "crs"


def _move_files_to_output_folder(
    unwrapped_paths: List[Path], conncomp_paths: List[Path], output_directory: Path
):
    for unw_p, cc_p in zip(unwrapped_paths, conncomp_paths):
        # get all the associated header/conncomp files too
        unw_new_name = output_directory / unw_p.name
        cc_new_name = output_directory / cc_p.name
        logger.info(f"Moving {unw_p} and {cc_p} into {output_directory}")
        unw_p.rename(unw_new_name)
        cc_p.rename(cc_new_name)
