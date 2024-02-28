from __future__ import annotations

import datetime
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import opera_utils as oput
from opera_utils import get_dates
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Bbox, Filename, TropoModel, TropoType
from dolphin.utils import format_date_pair

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

    SNWE: tuple[float, float, float, float]
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

    reference_file: Sequence[Filename]
    """Sequence of filenames used as reference files."""

    secondary_file: Sequence[Filename]
    """Sequence of filenames used as secondary files."""

    interferogram: str
    """Identifier for the interferogram."""

    reference_time: datetime.datetime
    """The reference image time."""

    secondary_time: datetime.datetime
    """The secondary image time."""


def estimate_tropospheric_delay(
    ifg_file_list: Sequence[Path],
    slc_files: Mapping[tuple[datetime.datetime], Sequence[Filename]],
    troposphere_files: Mapping[tuple[datetime.datetime], Sequence[Filename]],
    geom_files: dict[str, Path],
    output_dir: Path,
    tropo_package: str,
    tropo_model: TropoModel,
    tropo_delay_type: TropoType,
    epsg: int,
    bounds: Bbox,
):
    """Estimate the tropospheric delay corrections for each interferogram.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        List of interferogram files.
    slc_files : Mapping[tuple[datetime.datetime], Sequence[Filename]]
        Dictionary of SLC files indexed by date.
    troposphere_files : Mapping[tuple[datetime.datetime], Sequence[Filename]]
        Dictionary of troposphere files indexed by date.
    geom_files : dict[str, Path]
        Dictionary of geometry files with height and incidence angle, or
        los_east and los_north.
    output_dir : Path
        Output directory.
    tropo_package : str
        Troposphere processing package ('pyaps' or 'raider').
    tropo_model : TropoModel
        Tropospheric model (ERA5, HRES, ...).
    tropo_delay_type : TropoType
        Tropospheric delay type ('wet', 'hydrostatic', 'comb').
    epsg : int
        the EPSG code of the input data
    bounds : Bbox
        Output bounds.

    """
    # Read geogrid data
    xsize, ysize = io.get_raster_xysize(ifg_file_list[0])
    gt = io.get_raster_gt(ifg_file_list[0])
    ycoord, xcoord = oput.create_yx_arrays(gt, (ysize, xsize))  # 500 m spacing

    if epsg != 4326:
        left, bottom, right, top = transform_bounds(
            CRS.from_epsg(epsg), CRS.from_epsg(4326), *bounds
        )
    else:
        left, bottom, right, top = bounds

    tropo_height_levels = np.concatenate(([-100], np.arange(0, 9000, 500)))

    for key in slc_files:
        if len(key) == 1:
            first_date = key
            break
    wavelength = oput.get_radar_wavelength(slc_files[first_date][0])

    tropo_run = compute_pyaps if tropo_package.lower() == "pyaps" else compute_raider

    # comb is short for the summation of wet and dry components
    if (tropo_delay_type.value == "hydrostatic") and (tropo_package == "raider"):
        delay_type = "hydro"
    elif (tropo_delay_type.value == "hydrostatic") and (tropo_package == "pyaps"):
        delay_type = "dry"
    else:
        delay_type = tropo_delay_type.value

    output_tropo_dir = output_dir / "troposphere"
    output_tropo_dir.mkdir(exist_ok=True)
    for ifg in ifg_file_list:
        ref_date, sec_date = get_dates(ifg)

        date_str = format_date_pair(ref_date, sec_date)
        name = f"{date_str}_tropoDelay_pyaps_{tropo_model.value}_LOS_{delay_type}.tif"
        tropo_delay_product_path = output_tropo_dir / name

        if tropo_delay_product_path.exists():
            logger.info(f"{tropo_delay_product_path} exists, skipping")
            continue

        reference_date = (ref_date,)
        secondary_date = (sec_date,)

        if (
            reference_date not in troposphere_files
            or secondary_date not in troposphere_files
        ):
            logger.warning(f"Weather-model files do not exist for {date_str}, skipping")
            continue

        secondary_time = oput.get_zero_doppler_time(slc_files[secondary_date][0])
        if len(slc_files[reference_date]) == 0:
            # this is for when we have compressed slcs but the actual
            # reference date does not exist in the input data
            reference_time = datetime.datetime.combine(
                reference_date[0].date(), secondary_time.time()
            )
        else:
            reference_time = oput.get_zero_doppler_time(slc_files[reference_date][0])

        delay_parameters = DelayParams(
            x_coordinates=xcoord,
            y_coordinates=ycoord,
            z_coordinates=tropo_height_levels,
            SNWE=(bottom, top, left, right),
            epsg=epsg,
            tropo_model=tropo_model.value,
            delay_type=delay_type,
            wavelength=wavelength,
            shape=(ysize, xsize),
            geotransform=gt,
            reference_file=troposphere_files[reference_date],
            secondary_file=troposphere_files[secondary_date],
            reference_time=reference_time,
            secondary_time=secondary_time,
            interferogram=format_date_pair(ref_date, sec_date),
        )

        delay_datacube = tropo_run(delay_parameters)

        tropo_delay_2d = compute_2d_delay(delay_parameters, delay_datacube, geom_files)

        # Write 2D tropospheric correction layer to disc
        io.write_arr(
            arr=tropo_delay_2d,
            output_name=tropo_delay_product_path,
            like_filename=ifg,
        )

    return


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
    return np.ma.masked_invalid(tropo_delay_datacube)


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
    return tropo_delay_interpolator(pnts).reshape(hv.shape)


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
    dem_file = Path(geo_files["height"])

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
    # Use the same suffix for warping
    target_ds = gdal.Warp(
        "",  # Output to memory to read
        fspath(dem_file.resolve()),
        options=options,
    )

    dem = target_ds.ReadAsArray()

    if "los_east" in geo_files:
        # ISCE3 geocoded products
        los_east = io.load_gdal(geo_files["los_east"])
        los_north = io.load_gdal(geo_files["los_north"])
        los_up = np.sqrt(1 - los_east**2 - los_north**2)
    else:
        # ISCE2 radar coordinate
        los_up = np.cos(np.deg2rad(io.load_gdal(geo_files["incidence_angle"])))

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
