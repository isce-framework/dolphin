from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import opera_utils as oput
from numpy.typing import ArrayLike
from opera_utils import get_dates
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from scipy import interpolate
from scipy.ndimage import zoom

from dolphin import io
from dolphin._types import Bbox, Filename
from dolphin.timeseries import ReferencePoint
from dolphin.utils import format_date_pair

logger = logging.getLogger("dolphin")

###########

__all__ = ["estimate_ionospheric_delay"]

# constants
K = 40.31
SPEED_OF_LIGHT = 299792458  # meters per second
EARTH_RADIUS = 6371.0088e3  # km


def estimate_ionospheric_delay(
    ifg_file_list: Sequence[Path],
    slc_files: Mapping[tuple[datetime.datetime], Sequence[Filename]],
    tec_files: Mapping[tuple[datetime.datetime], Sequence[Filename]],
    geom_files: dict[str, Path],
    reference_point: ReferencePoint | None,
    output_dir: Path,
    epsg: int,
    bounds: Bbox,
) -> list[Path]:
    """Estimate the range delay (in meters) caused by ionosphere for each interferogram.

    Parameters
    ----------
    ifg_file_list : list[Path]
        List of interferogram files.
    slc_files : Dict[datetime.date, list[Filename]]
        Dictionary of SLC files indexed by date.
    tec_files : Dict[datetime.date, list[Filename]]
        Dictionary of TEC files indexed by date.
    geom_files : list[Path]
        Dictionary of geometry files with height and incidence angle, or
        los_east and los_north.
    reference_point : tuple[int, int], optional
        Reference point (row, col) used during time series inversion.
        If not provided, no spatial referencing is performed.
    output_dir : Path
        Output directory.
    epsg : int
        the EPSG code of the input data
    bounds : Bbox
        Output bounds.

    Returns
    -------
    list[Path]
        List of newly created ionospheric phase delay corrections.

    """
    if epsg != 4326:
        left, bottom, right, top = transform_bounds(
            CRS.from_epsg(epsg), CRS.from_epsg(4326), *bounds
        )
    else:
        left, bottom, right, top = bounds

    # Read the incidence angle
    if "los_east" in geom_files:
        # ISCE3 geocoded products
        los_east = io.load_gdal(geom_files["los_east"])
        los_north = io.load_gdal(geom_files["los_north"])
        inc_angle = np.arccos(np.sqrt(1 - los_east**2 - los_north**2)) * 180 / np.pi
    else:
        # ISCE2 radar coordinate
        inc_angle = io.load_gdal(geom_files["incidence_angle"])

    iono_inc_angle = incidence_angle_ground_to_iono(inc_angle)

    # frequency
    for key in slc_files:
        if "compressed" not in str(slc_files[key][0]).lower():
            one_of_slcs = slc_files[key][0]
            break

    wavelength = oput.get_radar_wavelength(one_of_slcs)
    freq = SPEED_OF_LIGHT / wavelength

    output_iono = output_dir / "ionosphere"
    output_iono.mkdir(exist_ok=True)

    output_paths: list[Path] = []

    num_lats, num_lons = iono_inc_angle.shape

    downsample_factor: int = 10
    # Create downsampled grid for efficiency
    num_lats, num_lons = iono_inc_angle.shape
    ds_num_lats = max(5, num_lats // downsample_factor)
    ds_num_lons = max(5, num_lons // downsample_factor)

    # Create the downsampled grid
    ds_out_lats = np.linspace(top, bottom, ds_num_lats)
    ds_out_lons = np.linspace(left, right, ds_num_lons)
    ds_lat_grid, ds_lon_grid = np.meshgrid(ds_out_lats, ds_out_lons, indexing="ij")

    # Downsample the incidence angle
    ds_inc_angle = zoom(
        iono_inc_angle, (ds_num_lats / num_lats, ds_num_lons / num_lons), order=1
    )
    # Ensure same size as the lat/lon grids
    ds_inc_angle = ds_inc_angle[:ds_num_lats, :ds_num_lons]

    for ifg in ifg_file_list:
        ref_date, sec_date = get_dates(ifg)

        name = f"{format_date_pair(ref_date, sec_date)}_ionoDelay.tif"
        iono_delay_product_name = output_iono / name

        output_paths.append(iono_delay_product_name)
        if iono_delay_product_name.exists():
            logger.info(
                "Tropospheric correction for interferogram "
                f"{format_date_pair(ref_date, sec_date)} already exists, skipping"
            )
            continue

        # The keys in slc_files do not necessarily have one date,
        # it means with the production file naming convention,
        # there will be multiple dates stored in the keys from get_dates
        # function. So in the following if we set reference_date = (ref_date, ),
        # there will be an error that it does not find the key.
        # Examples of the file naming convention for CSLC and compressed cslc is:
        # OPERA_L2_CSLC-S1_T042-088905-IW1\
        #                       _20221119T000000Z_20221120T000000Z_S1A_VV_v1.0.h5
        # OPERA_L2_COMPRESSED-CSLC-S1_T042-088905-IW1_20221107T000000Z_\
        #           20221107T000000Z_20230506T000000Z_20230507T000000Z_VV_v1.0.h5
        reference_date = next(key for key in slc_files if ref_date in key)
        # temporary fix for compressed SLCs while the required metadata i not included
        # in them. there will be modification in a future PR to add metadata to
        # compressed SLCs
        secondary_date = next(
            key
            for key in slc_files
            if sec_date in key and "compressed" not in str(slc_files[key][0]).lower()
        )

        secondary_time = oput.get_zero_doppler_time(slc_files[secondary_date][0])
        if "compressed" in str(slc_files[reference_date][0]).lower():
            # this is for when we have compressed slcs but the actual
            # reference date does not exist in the input data
            reference_time = secondary_time
        else:
            reference_time = oput.get_zero_doppler_time(slc_files[reference_date][0])

        # Calculate interpolated range delays
        vtec_reference = read_zenith_tec(
            reference_time, tec_files[(ref_date,)][0], ds_lat_grid, ds_lon_grid
        )
        # Make the downsampled (ds_) version
        ds_range_delay_reference = vtec_to_range_delay(
            vtec_reference, ds_inc_angle, freq
        )
        vtec_secondary = read_zenith_tec(
            secondary_time, tec_files[(sec_date,)][0], ds_lat_grid, ds_lon_grid
        )
        ds_range_delay_secondary = vtec_to_range_delay(
            vtec_secondary, ds_inc_angle, freq
        )

        # For displacement output, positive corresponds to motion toward the satellite
        # If range_delay_secondary is smaller than range_delay_reference, then the
        # resulting delay difference is positive. A smaller excess secondary delay
        # looks the same as motion toward the satellite.
        ds_ifg_iono_range_delay = ds_range_delay_reference - ds_range_delay_secondary

        # Finally upsample the end result:
        ifg_iono_range_delay = zoom(
            ds_ifg_iono_range_delay,
            (num_lats / ds_num_lats, num_lons / ds_num_lons),
            order=1,
        )

        if reference_point is not None:
            ref_row, ref_col = reference_point
            ifg_iono_range_delay -= ifg_iono_range_delay[ref_row, ref_col]

        # Write 2D tropospheric correction layer to disc
        io.write_arr(
            arr=ifg_iono_range_delay,
            output_name=iono_delay_product_name,
            like_filename=ifg,
            units="meters",
        )

    return output_paths


def incidence_angle_ground_to_iono(inc_angle: np.ndarray, iono_height: float = 450e3):
    """Calibrate incidence angle on the ground surface to the ionosphere shell.

    Equation (11) in Yunjun et al. (2022, TGRS)

    Parameters
    ----------
    inc_angle: np.ndarray
        incidence angle (in degrees) on the ground
    iono_height: float
        effective ionosphere height in meters under the thin-shell assumption

    Returns
    -------
    np.ndarray
        incidence angle (in degrees) on the iono shell

    """
    # convert degrees to radians
    inc_angle_rad = inc_angle * np.pi / 180

    inc_angle_iono = np.arcsin(
        EARTH_RADIUS * np.sin(inc_angle_rad) / (EARTH_RADIUS + iono_height)
    )
    inc_angle_iono *= 180.0 / np.pi

    return inc_angle_iono


def read_zenith_tec(
    dt: datetime.datetime, tec_file: Filename, lat: ArrayLike, lon: ArrayLike
) -> np.ndarray:
    """Read `tec_file` and interpolate zenith TEC for some latitudes and longitudes.

    Parameters
    ----------
    dt: datetime
        datetime of the acquisition
    tec_file: Filename
        path to the tec file corresponding to slc date
    lat: ArrayLike
        Latitude(s) at which to calculate zenith TEC
    lon: ArrayLike
        Longitude(s) at which to calculate zenith TEC

    Returns
    -------
    np.ndarray
        zenith TEC in TECU

    """
    utc_seconds = (dt.hour * 3600.0) + (dt.minute * 60.0) + dt.second
    return get_ionex_value(tec_file=tec_file, utc_sec=utc_seconds, lat=lat, lon=lon)


def vtec_to_range_delay(vtec: float, inc_angle: np.ndarray, freq: float):
    """Calculate/predict the range delay in SAR from TEC in zenith direction.

    Equation (6-11) from Yunjun et al. (2022).

    Parameters
    ----------
    vtec: float
        zenith TEC in TECU
    inc_angle: np.ndarray
        incidence angle at the ionospheric shell in deg
    freq: float
        radar carrier frequency in Hz.

    Returns
    -------
    np.ndarray
        predicted range delay in meters

    """
    # ignore no-data value in inc_angle
    if isinstance(inc_angle, np.ndarray):
        inc_angle[inc_angle == 0] = np.nan

    # convert to TEC in LOS based on equation (3) in Chen and Zebker (2012)

    # Equation (26) in Bohm & Schuh (2013) Chapter 2.
    n_iono_group = 1 + K * vtec * 1e16 / freq**2

    # Equation (8) in Yunjun et al. (2022, TGRS)
    ref_angle = (
        np.arcsin(1 / n_iono_group * np.sin(inc_angle * np.pi / 180)) * 180 / np.pi
    )

    tec = vtec / np.cos(ref_angle * np.pi / 180.0)

    # calculate range delay based on equation (1) in Chen and Zebker (2012)
    range_delay = (tec * 1e16 * K / (freq**2)).astype(np.float32)

    return range_delay


def get_ionex_value(
    tec_file: Filename, utc_sec: float, lat: ArrayLike, lon: ArrayLike
) -> np.ndarray:
    """Get the TEC value from input IONEX file for the input lat/lon/datetime.

    Parameters
    ----------
    tec_file: str
        Path of local TEC file
    utc_sec: float or 1D np.ndarray
        UTC time of the day in seconds
    lat: ArrayLike (float or np.ndarray)
        Latitude in degrees
    lon: ArrayLike (float or np.ndarray)
        Longitude in degrees

    Returns
    -------
    tec_val: np.ndarray
        Vertical TEC value in TECU

    Notes
    -----
    If passing arrays for `lat` and `lon`, they must be the same shape.

    Reference
    ---------
    Schaer, S., Gurtner, W., & Feltens, J. (1998). IONEX: The ionosphere map
    exchange format version 1.1. Paper presented at the Proceedings of the IGS AC
    workshop, Darmstadt, Germany.

    """
    # Get the time of the day in minutes
    utc_min = utc_sec / 60.0

    mins, lats, lons, tec_maps = read_ionex(tec_file)

    # Interpolate between consecutive TEC maps
    tec_val = interpolate.interpn(
        points=(mins, lats, lons),
        values=tec_maps,
        xi=(utc_min, lat, lon),
        bounds_error=False,
        method="linear",
    )
    return tec_val


def read_ionex(
    tec_file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read Total Electron Content (TEC) file in IONEX format.

    Parameters
    ----------
    tec_file: str
        Path to the TEC file in IONEX format

    Returns
    -------
    mins: np.ndarray
        1D array with time of the day in minutes
    lats: np.ndarray
        1D array with latitude in degrees
    lons: np.ndarray
        1D array with longitude in degrees
    tec_maps: np.ndarray
        3D array with vertical TEC in TECU


    Notes
    -----
    The order of `mins` is ascending:
    [   0.,  120.,  240.,  ..., 1200., 1320., 1440.]
    The order of `lats` is descending, with 2.5 degree spacing
    [ 87.5,  85. ,  82.5,  80., ..., -80. , -82.5, -85. , -87.5]
    `lons` is ascending, with 5 degree spacing
    [-180., -175., -170., ...,  170.,  175., 180.]

    """

    # functions for parsing strings from ionex file
    # link: https://github.com/daniestevez/jupyter_notebooks/blob/master/IONEX.ipynb
    def parse_map(tec_map_str, key="TEC", exponent=-1):
        tec_map_str = re.split(f".*END OF {key} MAP", tec_map_str)[0]
        tec_map = [
            np.fromstring(x, sep=" ")
            for x in re.split(".*LAT/LON1/LON2/DLON/H\\n", tec_map_str)[1:]
        ]
        return np.stack(tec_map) * 10**exponent

    # read IONEX file
    with open(tec_file) as f:
        fc = f.read()

        # read header
        header = fc.split("END OF HEADER")[0].split("\n")
        for line in header:
            if line.strip().endswith("# OF MAPS IN FILE"):
                num_map = int(line.split()[0])
            elif line.strip().endswith("DLAT"):
                lat0, lat1, lat_step = (float(x) for x in line.split()[:3])
            elif line.strip().endswith("DLON"):
                lon0, lon1, lon_step = (float(x) for x in line.split()[:3])
            elif line.strip().endswith("EXPONENT"):
                exponent = float(line.split()[0])

        # spatial coordinates
        num_lat = (lat1 - lat0) // lat_step + 1
        num_lon = (lon1 - lon0) // lon_step + 1
        lats = np.arange(lat0, lat0 + num_lat * lat_step, lat_step)
        lons = np.arange(lon0, lon0 + num_lon * lon_step, lon_step)

        # time stamps in minutes
        min_step = 24 * 60 / (num_map - 1)
        mins = np.arange(0, num_map * min_step, min_step)

        # read TEC maps
        tec_maps = np.array(
            [
                parse_map(t, key="TEC", exponent=exponent)
                for t in fc.split("START OF TEC MAP")[1:]
            ],
            dtype=np.float32,
        )

    return mins, lats, lons, tec_maps
