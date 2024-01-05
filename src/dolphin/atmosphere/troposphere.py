from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import Optional

import numpy as np
import opera_utils as oput
from osgeo import gdal
from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator

from dolphin import io, stitching
from dolphin._dates import _format_date_pair, get_dates
from dolphin._log import get_log
from dolphin._types import Bbox, Filename, TropoModel, TropoType

logger = get_log(__name__)

###########

__all__ = ["estimate_tropospheric_delay"]


@dataclass
class DelayParams:
    """Parameters for estimating tropospheric delay corrections."""

    x_coordinates: np.ndarray
    """Array of X coordinates."""

    y_coordinates: np.ndarray
    """Array of Y coordinates."""

    z_coordinates: np.ndarray
    """Array of Z coordinates."""

    SNWE: Bbox
    """ Bounding box of the data in SNWE format of RAiDER/PYAPS."""

    shape: tuple[int, int]
    """Shape of the data, specified as a tuple of two integers."""

    epsg: int
    """EPSG code for the coordinate reference system."""

    geotransform: list[float]
    """Sequence of geotransformation parameters."""

    wavelength: float
    """Radar wavelength."""

    tropo_model: str
    """Model used for tropospheric correction."""

    delay_type: str
    """Type of tropospheric delay."""

    reference_file: list[Filename]
    """Sequence of filenames used as reference files."""

    secondary_file: list[Filename]
    """Sequence of filenames used as secondary files."""

    interferogram: str
    """Identifier for the interferogram."""

    reference_time: datetime.datetime
    """The reference image time."""

    secondary_time: datetime.datetime
    """The secondary image time."""


def estimate_tropospheric_delay(
    ifg_file_list: list[Path],
    slc_files: dict[tuple[datetime.datetime], list[Filename]],
    troposphere_files: dict[tuple[datetime.datetime], list[Filename]],
    geom_files: list[Path],
    dem_file: Optional[Path],
    output_dir: Path,
    tropo_package: str,
    tropo_model: TropoModel,
    tropo_delay_type: TropoType,
    strides: dict[str, int] = {"x": 1, "y": 1},
):
    """Estimate the tropospheric delay corrections for each interferogram.

    Parameters
    ----------
    ifg_file_list : list[Path]
        List of interferogram files.
    slc_files : Dict[datetime.date, list[Filename]]
        Dictionary of SLC files indexed by date.
    troposphere_files : Dict[datetime.date, list[Filename]]
        Dictionary of troposphere files indexed by date.
    geom_files : list[Path]
        List of geometry files.
    dem_file : Optional[Path]
        DEM file.
    output_dir : Path
        Output directory.
    tropo_package : str
        Troposphere processing package ('pyaps' or 'raider').
    tropo_model : TropoModel
        Tropospheric model (ERA5, HRES, ...).
    tropo_delay_type : TropoType
        Tropospheric delay type ('wet', 'hydrostatic', 'comb').
    strides : Dict[str, int], optional
        Strides for resampling, by default {"x": 1, "y": 1}.
    """
    # Read geogrid data
    xsize, ysize = io.get_raster_xysize(ifg_file_list[0])
    crs = io.get_raster_crs(ifg_file_list[0])
    gt = io.get_raster_gt(ifg_file_list[0])
    ycoord, xcoord = oput.create_yx_arrays(gt, (ysize, xsize))  # 500 m spacing
    epsg = crs.to_epsg()
    out_bounds = io.get_raster_bounds(ifg_file_list[0])

    # prepare geometry data
    logger.info("Prepare geometry files...")
    geometry_dir = output_dir / "geometry"
    geometry_files = prepare_geometry(
        geometry_dir=geometry_dir,
        geo_files=geom_files,
        matching_file=ifg_file_list[0],
        dem_file=dem_file,
        epsg=epsg,
        out_bounds=out_bounds,
        strides=strides,
    )

    tropo_height_levels = np.concatenate(([-100], np.arange(0, 9000, 500)))

    first_date = next(iter(slc_files))
    wavelength = oput.get_radar_wavelength(slc_files[first_date][0])

    if tropo_package.lower() == "pyaps":
        tropo_run = compute_pyaps
    else:
        tropo_run = compute_raider

    # comb is short for the summation of wet and dry components
    if (tropo_delay_type.value == "hydrostatic") and (tropo_package == "raider"):
        delay_type = "hydro"
    elif (tropo_delay_type.value == "hydrostatic") and (tropo_package == "pyaps"):
        delay_type = "dry"
    else:
        delay_type = tropo_delay_type.value

    output_tropo = output_dir / "troposphere"
    output_tropo.mkdir(exist_ok=True)
    for ifg in ifg_file_list:
        ref_date, sec_date = get_dates(ifg)

        tropo_delay_product_name = (
            fspath(output_tropo)
            + f"/{_format_date_pair(ref_date, sec_date)}_tropoDelay_pyaps_{tropo_model.value}_LOS_{delay_type}.tif"
        )

        if Path(tropo_delay_product_name).exists():
            logger.info(
                f"Tropospheric correction for interferogram {_format_date_pair(ref_date, sec_date)} already exists, skipping"
            )
            continue

        reference_date = (ref_date,)
        secondary_date = (sec_date,)

        if (
            reference_date not in troposphere_files.keys()
            or secondary_date not in troposphere_files.keys()
        ):
            logger.warning(
                f"Weather-model files do not exist for interferogram {_format_date_pair(ref_date, sec_date)}, skipping"
            )
            continue

        delay_parameters = DelayParams(
            x_coordinates=xcoord,
            y_coordinates=ycoord,
            z_coordinates=tropo_height_levels,
            SNWE=oput.get_snwe(epsg, out_bounds),
            epsg=epsg,
            tropo_model=tropo_model.value,
            delay_type=delay_type,
            wavelength=wavelength,
            shape=(ysize, xsize),
            geotransform=gt,
            reference_file=troposphere_files[reference_date],
            secondary_file=troposphere_files[secondary_date],
            reference_time=oput.get_zero_doppler_time(slc_files[reference_date][0]),
            secondary_time=oput.get_zero_doppler_time(slc_files[secondary_date][0]),
            interferogram=_format_date_pair(ref_date, sec_date),
        )

        delay_datacube = tropo_run(delay_parameters)

        tropo_delay_2d = compute_2d_delay(
            delay_parameters, delay_datacube, geometry_files
        )

        # Write 2D tropospheric correction layer to disc
        io.write_arr(
            arr=tropo_delay_2d,
            output_name=tropo_delay_product_name,
            like_filename=ifg,
        )

    return


def prepare_geometry(
    geometry_dir: Path,
    geo_files: list[Path],
    matching_file: Path,
    dem_file: Optional[Path],
    epsg: int,
    out_bounds: Bbox,
    strides: dict[str, int] = {"x": 1, "y": 1},
) -> dict[str, Path]:
    """Prepare geometry files.

    Parameters
    ----------
    geometry_dir : Path
        Output directory for geometry files.
    geo_files : list[Path]
        list of geometry files.
    matching_file : Path
        Matching file.
    dem_file : Optional[Path]
        DEM file.
    epsg : int
        EPSG code.
    out_bounds : Bbox
        Output bounds.
    strides : Dict[str, int], optional
        Strides for resampling, by default {"x": 1, "y": 1}.

    Returns
    -------
    Dict[str, Path]
        Dictionary of prepared geometry files.
    """
    geometry_dir.mkdir(exist_ok=True)

    stitched_geo_list = {}

    # local_incidence_angle needed by anyone?
    datasets = ["los_east", "los_north"]

    for ds_name in datasets:
        outfile = geometry_dir / f"{ds_name}.tif"
        logger.info(f"Creating {outfile}")
        stitched_geo_list[ds_name] = outfile
        ds_path = f"/data/{ds_name}"
        cur_files = [io.format_nc_filename(f, ds_name=ds_path) for f in geo_files]

        no_data = 0

        stitching.merge_images(
            cur_files,
            outfile=outfile,
            driver="GTiff",
            out_bounds=out_bounds,
            out_bounds_epsg=epsg,
            in_nodata=no_data,
            out_nodata=no_data,
            target_aligned_pixels=True,
            strides=strides,
            resample_alg="nearest",
            overwrite=False,
        )

    if dem_file:
        height_file = geometry_dir / "height.tif"
        stitched_geo_list["height"] = height_file
        if not height_file.exists():
            logger.info(f"Creating {height_file}")
            stitching.warp_to_match(
                input_file=dem_file,
                match_file=matching_file,
                output_file=height_file,
                resample_alg="cubic",
            )

    return stitched_geo_list


def compute_pyaps(delay_parameters: DelayParams) -> np.ndarray:
    """Compute tropospheric delay datacube using PyAPS.

    Parameters
    ----------
    delay_parameters : DelayParams
        delay parameters and grid information.

    Returns
    -------
    np.ndarray
       tropospheric delay datacube.
    """
    import pyaps3 as pa

    # X and y for the entire datacube
    y_2d, x_2d = np.meshgrid(
        delay_parameters.y_coordinates, delay_parameters.x_coordinates, indexing="ij"
    )

    # Lat/lon coordinates
    lat_datacube, lon_datacube = oput.transform_xy_to_latlon(
        delay_parameters.epsg, x_2d, y_2d
    )

    tropo_delay_datacube_list = []

    for hgt in delay_parameters.z_coordinates:
        dem_datacube = np.full(lat_datacube.shape, hgt)

        # Delay for the reference image
        ref_aps_estimator = pa.PyAPS(
            fspath(delay_parameters.reference_file[0]),
            dem=dem_datacube,
            inc=0.0,
            lat=lat_datacube,
            lon=lon_datacube,
            grib=delay_parameters.tropo_model,
            humidity="Q",
            model=delay_parameters.tropo_model,
            verb=False,
            Del=delay_parameters.delay_type,
        )

        phs_ref = ref_aps_estimator.getdelay()

        # Delay for the secondary image
        second_aps_estimator = pa.PyAPS(
            fspath(delay_parameters.secondary_file[0]),
            dem=dem_datacube,
            inc=0.0,
            lat=lat_datacube,
            lon=lon_datacube,
            grib=delay_parameters.tropo_model,
            humidity="Q",
            model=delay_parameters.tropo_model,
            verb=False,
            Del=delay_parameters.delay_type,
        )

        phs_second = second_aps_estimator.getdelay()

        # Convert the delay in meters to radians
        tropo_delay_datacube_list.append(
            -(phs_ref - phs_second) * 4.0 * np.pi / delay_parameters.wavelength
        )

    # Tropo delay datacube
    tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
    # Create a maksed datacube that excludes the NaN values
    tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

    return tropo_delay_datacube_masked


def compute_raider(delay_parameters: DelayParams) -> np.ndarray:
    """Compute tropospheric delay datacube using RAiDER.

    Parameters
    ----------
    delay_parameters : DelayParams
        delay parameters and grid information.

    Returns
    -------
    np.ndarray
       tropospheric delay datacube.
    """
    from RAiDER.delay import tropo_delay as raider_tropo_delay
    from RAiDER.llreader import BoundingBox
    from RAiDER.losreader import Zenith

    reference_weather_model_file = delay_parameters.reference_file
    secondary_weather_model_file = delay_parameters.secondary_file

    aoi = BoundingBox(delay_parameters.SNWE)
    aoi.xpts = delay_parameters.x_coordinates
    aoi.ypts = delay_parameters.y_coordinates

    # Zenith
    delay_direction_obj = Zenith()

    # Troposphere delay computation
    # Troposphere delay datacube computation
    tropo_delay_reference, _ = raider_tropo_delay(
        dt=delay_parameters.reference_time,
        weather_model_file=reference_weather_model_file,
        aoi=aoi,
        los=delay_direction_obj,
        height_levels=delay_parameters.z_coordinates,
        out_proj=delay_parameters.epsg,
    )

    tropo_delay_secondary, _ = raider_tropo_delay(
        dt=delay_parameters.secondary_time,
        weather_model_file=secondary_weather_model_file,
        aoi=aoi,
        los=delay_direction_obj,
        height_levels=delay_parameters.z_coordinates,
        out_proj=delay_parameters.epsg,
    )

    # Compute troposphere delay with raider package
    # comb is the summation of wet and hydro components
    if delay_parameters.delay_type == "comb":
        tropo_delay = (
            tropo_delay_reference["wet"]
            + tropo_delay_reference["hydro"]
            - tropo_delay_secondary["wet"]
            - tropo_delay_secondary["hydro"]
        )
    else:
        tropo_delay = (
            tropo_delay_reference[delay_parameters.delay_type]
            - tropo_delay_secondary[delay_parameters.delay_type]
        )

    # Convert it to radians units
    tropo_delay_datacube = -tropo_delay * 4.0 * np.pi / delay_parameters.wavelength

    # Create a masked datacube that excludes the NaN values
    tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

    # Interpolate to radar grid to keep its dimension consistent with other datacubes
    tropo_delay_interpolator = RegularGridInterpolator(
        (tropo_delay_reference.z, tropo_delay_reference.y, tropo_delay_reference.x),
        tropo_delay_datacube_masked,
        method="linear",
        bounds_error=False,
    )

    # Interpolate the troposphere delay
    hv, yv, xv = np.meshgrid(
        delay_parameters.z_coordinates,
        delay_parameters.y_coordinates,
        delay_parameters.x_coordinates,
        indexing="ij",
    )

    pnts = np.stack((hv.flatten(), yv.flatten(), xv.flatten()), axis=-1)

    # Interpolate
    tropo_delay_datacube = tropo_delay_interpolator(pnts).reshape(hv.shape)

    return tropo_delay_datacube


def compute_2d_delay(
    delay_parameters: DelayParams,
    delay_datacube: np.ndarray,
    geo_files: dict[str, Path],
) -> np.ndarray:
    """Compute 2D delay.

    Parameters
    ----------
    delay_parameters : DelayParams
        dataclass containing tropospheric delay data.

    delay_datacube : np.ndarray
        delay datacube for the x,y,z coordinates in delay_parameters

    geo_files : dict[str, Path]
        Dictionary containing paths to geospatial files.

    Returns
    -------
    np.ndarray
        Computed 2D delay.
    """
    dem_file = geo_files["height"]

    ysize, xsize = delay_parameters.shape
    x_origin, x_res, x_, y_origin, y_, y_res = delay_parameters.geotransform

    x = 0
    y = 0
    left = x_origin + x * x_res + y * x_
    top = y_origin + x * y_ + y * y_res

    x = xsize
    y = ysize

    right = x_origin + x * x_res + y * x_
    bottom = y_origin + x * y_ + y * y_res

    bounds = (left, bottom, right, top)

    crs = CRS.from_epsg(delay_parameters.epsg)

    options = gdal.WarpOptions(
        dstSRS=crs,
        format="MEM",
        xRes=x_res,
        yRes=y_res,
        outputBounds=bounds,
        outputBoundsSRS=crs,
        resampleAlg="near",
    )
    target_ds = gdal.Warp(
        os.path.abspath(fspath(dem_file) + ".temp"),
        os.path.abspath(fspath(dem_file)),
        options=options,
    )

    dem = target_ds.ReadAsArray()

    los_east = io.load_gdal(geo_files["los_east"])
    los_north = io.load_gdal(geo_files["los_north"])
    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    mask = los_east > 0

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin, x_origin + x_res * xsize, x_res)

    yv, xv = np.meshgrid(y, x, indexing="ij")

    tropo_delay_interpolator = RegularGridInterpolator(
        (
            delay_parameters.z_coordinates,
            delay_parameters.y_coordinates,
            delay_parameters.x_coordinates,
        ),
        delay_datacube,
        method="linear",
        bounds_error=False,
    )

    tropo_delay_2d = np.zeros(dem.shape, dtype=np.float32)

    nline = 100
    for i in range(0, dem.shape[1], 100):
        if i + 100 > dem.shape[0]:
            nline = dem.shape[0] - i
        pnts = np.stack(
            (
                dem[i : i + 100, :].flatten(),
                yv[i : i + 100, :].flatten(),
                xv[i : i + 100, :].flatten(),
            ),
            axis=-1,
        )
        tropo_delay_2d[i : i + 100, :] = tropo_delay_interpolator(pnts).reshape(
            nline, dem.shape[1]
        )

    return (tropo_delay_2d / los_up) * mask
