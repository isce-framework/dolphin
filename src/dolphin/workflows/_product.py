"""Module for creating the OPERA output product in NetCDF format."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyproj
from numpy.typing import ArrayLike

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)


BASE_GROUP = "/science/SENTINEL1"
DISP_GROUP = f"{BASE_GROUP}/DISP"
QUALITY_GROUP = f"{DISP_GROUP}/quality"
CORRECTIONS_GROUP = f"{DISP_GROUP}/corrections"
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
HDF5_OPTS = io.DEFAULT_HDF5_OPTIONS.copy()
HDF5_OPTS["chunks"] = tuple(HDF5_OPTS["chunks"])  # type: ignore


# Make a class holding the dataset names/attrs
# so we can use it in the create_dataset calls
@dataclass
class DatasetInfo:
    """Convenience class to create a dataset in the output product."""

    name: str
    data: ArrayLike
    description: str
    fillvalue: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    xy_scales: Optional[Tuple[h5py.Dataset, h5py.Dataset]] = None

    def __post_init__(self):
        self.attrs.update(long_name=self.description)
        if self.fillvalue is not None:
            self.attrs["_FillValue"] = self.fillvalue

    def create(
        self,
        group: h5py.Group,
    ):
        """Create the dataset in the given h5py.Group."""
        dset = group.create_dataset(
            self.name,
            data=self.data,
            fillvalue=self.fillvalue,
            **HDF5_OPTS,
        )
        if self.xy_scales is not None:
            self._attach_scales(dset, *self.xy_scales)
        dset.attrs.update(self.attrs)

    def _attach_scales(
        self, dset: h5py.Dataset, x_ds: h5py.Dataset, y_ds: h5py.Dataset
    ):
        # Attach the X/Y coordinates
        self.attrs["grid_mapping"] = GRID_MAPPING_DSET
        dset.dims[0].attach_scale(y_ds)
        dset.dims[1].attach_scale(x_ds)


class ConnCompDatasetInfo(DatasetInfo):
    """Class for the connected components dataset."""

    def __init__(self, data: ArrayLike, xy_scales: Tuple[h5py.Dataset, h5py.Dataset]):
        super().__init__(
            name="connected_components",
            data=data,
            description="Connected components of the unwrapped phase",
            fillvalue=0,
            attrs=dict(units="unitless"),
            xy_scales=xy_scales,
        )


class UnwrappedDatasetInfo(DatasetInfo):
    """Class for the unwrapped phase dataset."""

    def __init__(self, data: ArrayLike, xy_scales: Tuple[h5py.Dataset, h5py.Dataset]):
        super().__init__(
            name="unwrapped_phase",
            data=data,
            description="Unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
            xy_scales=xy_scales,
        )


class TcorrDatasetInfo(DatasetInfo):
    """Class for the temporal correlation dataset."""

    def __init__(self, data: ArrayLike, xy_scales: Tuple[h5py.Dataset, h5py.Dataset]):
        super().__init__(
            name="temporal_correlation",
            data=data,
            description="Temporal correlation of phase inversion",
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
            xy_scales=xy_scales,
        )


class TropoDatasetInfo(DatasetInfo):
    """Class for the tropospheric delay dataset."""

    def __init__(self, data: ArrayLike, xy_scales: Tuple[h5py.Dataset, h5py.Dataset]):
        super().__init__(
            name="tropospheric_delay",
            data=data,
            description="Tropospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
            xy_scales=xy_scales,
        )


class IonosphereDatasetInfo(DatasetInfo):
    """Class for the ionospheric delay dataset."""

    def __init__(self, data: ArrayLike, xy_scales: Tuple[h5py.Dataset, h5py.Dataset]):
        super().__init__(
            name="ionospheric_delay",
            data=data,
            description="Ionospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
            xy_scales=xy_scales,
        )


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

    with h5py.File(output_name, "w") as f:
        # Create the NetCDF file
        f.attrs.update(GLOBAL_ATTRS)

        # Create the '/science/SENTINEL1/DISP/grids/displacement' group
        displacement_group = f.create_group(DISP_GROUP)
        quality_group = f.create_group(QUALITY_GROUP)

        # Set up the grid mapping variable
        _create_grid_mapping(displacement_group, crs, gt)

        # Set up the X/Y variables
        x_ds, y_ds = _create_xy_dsets(displacement_group, gt, unw_arr.shape)

        # Write the displacement array / conncomp arrays
        UnwrappedDatasetInfo(unw_arr, (x_ds, y_ds)).create(displacement_group)

        ConnCompDatasetInfo(conncomp_arr, (x_ds, y_ds)).create(quality_group)
        TcorrDatasetInfo(tcorr_arr, (x_ds, y_ds)).create(quality_group)

        # Create the '/science/SENTINEL1/DISP/corrections' group
        corrections_group = f.create_group(CORRECTIONS_GROUP)
        ionosphere = corrections.get("ionosphere")
        if ionosphere is not None:
            IonosphereDatasetInfo(ionosphere, (x_ds, y_ds)).create(corrections_group)

        troposphere = corrections.get("troposphere")
        if troposphere is not None:
            TropoDatasetInfo(troposphere, (x_ds, y_ds)).create(corrections_group)


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
