"""Module for creating the OPERA output product in NetCDF format."""
from typing import Any, Dict, List, Optional, Tuple

import h5netcdf
import numpy as np
import pyproj
from numpy.typing import ArrayLike

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)


BASE_GROUP = "/science/SENTINEL1"
DISP_GROUP = f"{BASE_GROUP}/DISP"
CORRECTIONS_GROUP = f"{DISP_GROUP}/corrections"
GLOBAL_ATTRS = dict(
    Conventions="CF-1.8",
    contact="operaops@jpl.nasa.gov",
    institution="NASA JPL",
    mission_name="OPERA",
    reference_document="TBD",
    title="OPERA L3_DISP_S1 Product",
)

# Convert chunks to a tuple or h5py errors
HDF5_OPTS = io.DEFAULT_HDF5_OPTIONS.copy()
HDF5_OPTS["chunks"] = tuple(HDF5_OPTS["chunks"])  # type: ignore
# The GRID_MAPPING_DSET variable is used to store the name of the dataset containing
# the grid mapping information, which includes the coordinate reference system (CRS)
# and the GeoTransform. This is in accordance with the CF 1.8 conventions for adding
# geospatial metadata to NetCDF files.
# http://cfconventions.org/cf-conventions/cf-conventions.html#grid-mappings-and-projections
# Note that the name "spatial_ref" used here is arbitrary, but it follows the default
# used by other libraries, such as rioxarray:
# https://github.com/corteva/rioxarray/blob/5783693895b4b055909c5758a72a5d40a365ef11/rioxarray/rioxarray.py#L34 # noqa
GRID_MAPPING_DSET = "spatial_ref"


def _create_dataset(
    *,
    group: h5netcdf.Group,
    name: str,
    data: np.ndarray,
    description: str,
    fillvalue: Optional[float],
    attrs: Optional[Dict[str, Any]],
) -> h5netcdf.Variable:
    if attrs is None:
        attrs = {}
    attrs.update(long_name=description)

    dset = group.create_variable(
        name,
        ["y_coordinate", "x_coordinate"],
        data=data,
        fillvalue=fillvalue,
        **HDF5_OPTS,
    )
    dset.attrs.update(attrs)
    return dset


def _create_geo_dataset(
    *,
    group: h5netcdf.Group,
    name: str,
    data: np.ndarray,
    description: str,
    fillvalue: Optional[float],
    attrs: Optional[Dict[str, Any]],
) -> h5netcdf.Variable:
    dset = _create_dataset(
        group=group,
        name=name,
        data=data,
        description=description,
        fillvalue=fillvalue,
        attrs=attrs,
    )
    dset.attrs["grid_mapping"] = GRID_MAPPING_DSET
    return dset


def create_output_product(
    unw_filename: Filename,
    conncomp_filename: Filename,
    tcorr_filename: Filename,
    output_name: Filename,
    corrections: Dict[str, ArrayLike] = {},
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
    unw_filename : Filename
        The path to the input unwrapped phase image.
    conncomp_filename : Filename
        The path to the input connected components image.
    tcorr_filename : Filename
        The path to the input temporal correlation image.
    output_name : Filename, optional
        The path to the output NetCDF file, by default "output.nc"
    corrections : Dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    """
    # Read the Geotiff file and its metadata
    crs = io.get_raster_crs(unw_filename)
    gt = io.get_raster_gt(unw_filename)
    unw_arr = io.load_gdal(unw_filename)

    conncomp_arr = io.load_gdal(conncomp_filename)
    tcorr_arr = io.load_gdal(tcorr_filename)

    # Get the nodata mask (which for snaphu is 0)
    mask = unw_arr == 0
    # Set to NaN for final output
    unw_arr[mask] = np.nan

    assert unw_arr.shape == conncomp_arr.shape == tcorr_arr.shape

    # with h5py.File(output_name, "w") as f:
    with h5netcdf.File(output_name, "w") as f:
        # Create the NetCDF file
        f.attrs.update(GLOBAL_ATTRS)

        # Create the '/science/SENTINEL1/DISP/grids/displacement' group
        displacement_group = f.create_group(DISP_GROUP)

        # Set up the grid mapping variable for each group with rasters
        _create_grid_mapping(group=displacement_group, crs=crs, gt=gt)

        # Set up the X/Y variables for each group
        _create_yx_dsets(group=displacement_group, gt=gt, shape=unw_arr.shape)
        # Write the displacement array / conncomp arrays
        _create_geo_dataset(
            group=displacement_group,
            name="unwrapped_phase",
            data=unw_arr,
            description="Unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )

        # scales2 = _create_yx_dsets(group=quality_group, gt=gt, shape=unw_arr.shape)
        _create_geo_dataset(
            group=displacement_group,
            name="connected_components",
            data=conncomp_arr,
            description="Connected components of the unwrapped phase",
            fillvalue=0,
            attrs=dict(units="unitless"),
        )
        _create_geo_dataset(
            group=displacement_group,
            name="temporal_correlation",
            data=tcorr_arr,
            description="Temporal correlation of phase inversion",
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
        )

        # Create the '/science/SENTINEL1/DISP/corrections' group
        corrections_group = f.create_group(CORRECTIONS_GROUP)

        troposphere = corrections.get("troposphere")
        _create_grid_mapping(group=corrections_group, crs=crs, gt=gt)
        _create_yx_dsets(group=corrections_group, gt=gt, shape=unw_arr.shape)
        if troposphere is not None:
            # TropoDatasetInfo(troposphere, scales).create(corrections_group)
            _create_geo_dataset(
                group=corrections_group,
                name="tropospheric_delay",
                data=troposphere,
                description=(
                    "Tropospheric phase delay used to correct the unwrapped phase"
                ),
                fillvalue=np.nan,
                attrs=dict(units="radians"),
            )
        ionosphere = corrections.get("ionosphere")
        if ionosphere is not None:
            _create_geo_dataset(
                group=corrections_group,
                name="ionospheric_delay",
                data=ionosphere,
                description=(
                    "Ionospheric phase delay used to correct the unwrapped phase"
                ),
                fillvalue=np.nan,
                attrs=dict(units="radians"),
            )


def _create_yx(
    gt: List[float], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    return y, x


def _create_yx_dsets(
    group: h5netcdf.Group,
    gt: List[float],
    shape: Tuple[int, int],
) -> Tuple[h5netcdf.Dataset, h5netcdf.Dataset]:
    """Create the x and y coordinate datasets."""
    y, x = _create_yx(gt, shape)

    if not group.dimensions:
        group.dimensions = dict(y_coordinate=y.size, x_coordinate=x.size)
    # Create the datasets
    y_ds = group.create_variable("y_coordinate", ("y_coordinate",), data=y, dtype=float)
    x_ds = group.create_variable("x_coordinate", ("x_coordinate",), data=x, dtype=float)

    for name, ds in zip(["y_coordinate", "x_coordinate"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} of projection"
        ds.attrs["units"] = "m"

    return y_ds, x_ds


def _create_grid_mapping(group, crs: pyproj.CRS, gt: List[float]) -> h5netcdf.Dataset:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_variable(GRID_MAPPING_DSET, (), data=0, dtype=int)

    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs["GeoTransform"] = gt_string
    return dset
