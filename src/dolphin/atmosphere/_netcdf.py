from __future__ import annotations

import os
import datetime
import numpy as np # type: ignore
from pyproj import CRS, Transformer # type: ignore
from typing import Union, Optional, Iterable
import xarray # type: ignore
from scipy.interpolate import RegularGridInterpolator as Interpolator # type: ignore
from collections import defaultdict
import itertools
try:
    import multiprocessing as mp
except ImportError:
    mp = None
    

from dolphin._log import get_log
from opera_utils import reproject_bounds # type: ignore
from dolphin._types import Filename, TropoModel

logger = get_log(__name__)

###########
# Mostly inherited from RAiDER

__all__ = ["delay_from_netcdf"]

def delay_from_netcdf(
        dt: datetime.datetime,
        weather_model_file: str,
        SNWE: tuple[float, float, float, float],
        height_levels: Optional[list[float]],
        out_proj: Union[int, str]=4326,
        weather_model: TropoModel=TropoModel.ECMWF,
    ):
    """Calculate integrated delays on query points.

    Parameters
    ----------
    dt: Datetime                 
        Datetime object for determining when to calculate delays
    weather_model_File: string  
        Name of the NETCDF file containing a pre-processed weather model
    aoi: AOI object             
        AOI object
    height_levels: list         
        (optional) list of height levels on which to calculate delays. Only needed for cube generation.
    out_proj: int,str           
        (optional) EPSG code for output projection

    Returns
    -------
        xarray Dataset *or* ndarrays:  
            wet and hydrostatic delays at the grid nodes / query points.
    """
    crs = CRS(out_proj)
    
    # Load CRS from weather model file
    with xarray.load_dataset(weather_model_file) as ds:
        try:
            wm_proj = CRS.from_wkt(ds['proj'].attrs['crs_wkt'])
        except KeyError:
            logger.warning("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
            wm_proj = CRS.from_epsg(4326)

        if 'hydro' in ds:
            wmodel_file = weather_model_file
        else:
            wmodel_file = _convert_NetCDF(weather_model_file=weather_model_file,
                                          lat_lon_bounds=SNWE,
                                          weather_model_output_dir='/tmp',
                                          weather_model=weather_model)
    
    # get heights
    with xarray.load_dataset(wmodel_file) as ds:
        wm_levels = ds.z.values
    
    if height_levels is None:
        height_levels = wm_levels


    #TODO: expose this as library function
    ds = _get_delays_on_cube(dt, wmodel_file, wm_proj, SNWE, height_levels, crs)

    
    return ds


def get_output_xygrid(SNWE: tuple[float, float, float, float], 
                      cube_spacing_m: float=500, 
                      dst_epsg:int=4326):
    """Reads the SNWE bounding box in lat/lon and creates the output grid.
    
    Parameters
    ----------
    SNWE: tuple[float, float, float, float]
        The bounding box in lat/lon coordinates
    cube_spacing_m: float
        The output grid spacing in meters
    dest_epsg:
        The output grid epsg code

    Returns
    -------
        The x and y coordinates of the output grid
    """
    out_snwe = SNWE
    if not dst_epsg == 4326:
        out_snwe = reproject_bounds(SNWE, src_epsg=4326, dst_epsg=dst_epsg)

    crs = CRS.from_epsg(dst_epsg)

    # Build the output grid
    if all(axis_info.unit_name == 'degree' for axis_info in crs.axis_info):
        out_spacing = cube_spacing_m / 1e5  
    else:
        out_spacing = cube_spacing_m
    
    xpts = np.arange(out_snwe[2], out_snwe[3] + out_spacing, out_spacing)
    ypts = np.arange(out_snwe[1], out_snwe[0] - out_spacing, -out_spacing)
    return xpts, ypts

def _get_delays_on_cube(dt, weather_model_file, wm_proj, SNWE, heights, crs, nproc=1):
    """Generate delays on cube"""
    
    zpts = np.array(heights)
    out_type = "zenith" 

    # Get ZTD interpolators
    try:
        ifWet, ifHydro = _getInterpolators(weather_model_file, "total")
    except RuntimeError:
        logger.exception('Failed to get weather model %s interpolators.', weather_model_file)


    # Build cube
    xpts, ypts = get_output_xygrid(SNWE=SNWE)
    wetDelay, hydroDelay = _build_cube(xpts, ypts, zpts, wm_proj, crs, [ifWet, ifHydro])

    if np.isnan(wetDelay).any() or np.isnan(hydroDelay).any():
        logger.critical('There are missing delay values. Check your inputs.')

    # Write output file
    ds = _writeResultsToXarray(dt, xpts, ypts, zpts, crs, wetDelay,
                              hydroDelay, weather_model_file, out_type)

    return ds

def _getInterpolators(wm_file, kind='pointwise', shared=False):
    '''
    Read 3D gridded data from a processed weather model file and wrap it with
    the scipy RegularGridInterpolator

    The interpolator grid is (y, x, z)
    '''
    # Get the weather model data
    try:
        ds = xarray.load_dataset(wm_file)
    except ValueError:
        ds = wm_file
    
    xs_wm = np.array(ds.variables['x'][:])
    ys_wm = np.array(ds.variables['y'][:])
    zs_wm = np.array(ds.variables['z'][:])

    wet = ds.variables['wet_total' if kind=='total' else 'wet'][:]
    hydro = ds.variables['hydro_total' if kind=='total' else 'hydro'][:]

    wet = np.array(wet).transpose(1, 2, 0)
    hydro = np.array(hydro).transpose(1, 2, 0)

    if np.any(np.isnan(wet)) or np.any(np.isnan(hydro)):
        logger.critical(f'Weather model contains NaNs!')

    # If shared interpolators are requested
    # The arrays are not modified - so turning off lock for performance
    if shared:
        xs_wm = _make_shared_raw(xs_wm)
        ys_wm = _make_shared_raw(ys_wm)
        zs_wm = _make_shared_raw(zs_wm)
        wet   = _make_shared_raw(wet)
        hydro = _make_shared_raw(hydro)

    
    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan, bounds_error = False)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan, bounds_error = False)

    return ifWet, ifHydro

def _make_shared_raw(inarr):
    """Make numpy view array of mp.Array."""
    
    # Create flat shared array
    if mp is None:
        raise ImportError('multiprocessing is not available')
    
    shared_arr = mp.RawArray('d', inarr.size)
    # Create a numpy view of it
    shared_arr_np = np.ndarray(inarr.shape, dtype=np.float64,
                               buffer=shared_arr)
    # Copy data to shared array
    np.copyto(shared_arr_np, inarr)

    return shared_arr_np


def _build_cube(xpts, ypts, zpts, model_crs, pts_crs, interpolators):
    """Iterate over interpolators and build a cube using Zenith."""
    # Create a regular 2D grid
    xx, yy = np.meshgrid(xpts, ypts)

    # Output arrays
    outputArrs = [np.zeros((zpts.size, ypts.size, xpts.size))
                  for mm in range(len(interpolators))]


    # Loop over heights and compute delays
    for ii, ht in enumerate(zpts):

        # pts is in weather model system;
        if model_crs != pts_crs:
            # lat / lon / height for hrrr
            pts = transformPoints(yy, xx, np.full(yy.shape, ht),
                                               pts_crs, model_crs)
        else:
            pts = np.stack([yy, xx, np.full(yy.shape, ht)], axis=-1)

        for mm, intp in enumerate(interpolators):
            outputArrs[mm][ii,...] = intp(pts)

    return outputArrs


def transformPoints(lats: np.ndarray, 
                    lons: np.ndarray, 
                    hgts: np.ndarray, 
                    old_proj: CRS, 
                    new_proj: CRS) -> np.ndarray:
    '''Transform lat/lon/hgt data to an array of points in a new projection

    Parameters
    ----------
        lats : ndarray   
            WGS-84 latitude (EPSG: 4326)
        lons : ndarray   
            ditto for longitude
        hgts : ndarray   
            Ellipsoidal height in meters
        old_proj: CRS   
            the original projection of the points
        new_proj: CRS   
            the new projection in which to return the points

    Returns:
        ndarray: the array of query points in the weather model coordinate system (YX)
    '''
    # Flags for flipping inputs or outputs
    if not isinstance(new_proj, CRS):
        new_proj = CRS.from_epsg(new_proj.lstrip('EPSG:'))
    if not isinstance(old_proj, CRS):
        old_proj = CRS.from_epsg(old_proj.lstrip('EPSG:'))

    t = Transformer.from_crs(old_proj, new_proj, always_xy=True)

    res  = t.transform(lons, lats, hgts)

    # lat/lon/height
    return  np.stack([res[1], res[0], res[2]], axis=-1)


def _writeResultsToXarray(dt:datetime.datetime, 
                          xpts: np.ndarray, 
                          ypts: np.ndarray, 
                          zpts: np.ndarray, 
                          crs: CRS, 
                          wetDelay: np.ndarray, 
                          hydroDelay: np.ndarray, 
                          weather_model_file: str, 
                          out_type: str):
    '''write a 1-D array to a NETCDF5 file.'''
    
    ds = xarray.Dataset(
        data_vars=dict(
            wet=(["z", "y", "x"],
                 wetDelay,
                 {"units" : "m",
                  "description": f"wet {out_type} delay",
                  # 'crs': crs.to_epsg(),
                  "grid_mapping": "crs",

                 }),
            hydro=(["z", "y", "x"],
                   hydroDelay,
                   {"units": "m",
                    # 'crs': crs.to_epsg(),
                    "description": f"hydrostatic {out_type} delay",
                    "grid_mapping": "crs",
                   }),
        ),
        coords=dict(
            x=(["x"], xpts),
            y=(["y"], ypts),
            z=(["z"], zpts),
        ),
        attrs=dict(
            Conventions="CF-1.7",
            title="Geo cube",
            source=os.path.basename(weather_model_file),
            history=str(datetime.datetime.now(datetime.timezone.utc)),
            description=f"Geo cube - {out_type}",
            reference_time=dt.strftime("%Y%m%dT%H:%M:%S"),
        ),
    )

    # Write projection system mapping
    ds["crs"] = int(-2147483647) # dummy placeholder
    for k, v in crs.to_cf().items():
        ds.crs.attrs[k] = v

    # Write z-axis information
    ds.z.attrs["axis"] = "Z"
    ds.z.attrs["units"] = "m"
    ds.z.attrs["description"] = "height above ellipsoid"

    # If in degrees
    if crs.axis_info[0].unit_name == "degree":
        ds.y.attrs["units"] = "degrees_north"
        ds.y.attrs["standard_name"] = "latitude"
        ds.y.attrs["long_name"] = "latitude"

        ds.x.attrs["units"] = "degrees_east"
        ds.x.attrs["standard_name"] = "longitude"
        ds.x.attrs["long_name"] = "longitude"

    else:
        ds.y.attrs["axis"] = "Y"
        ds.y.attrs["standard_name"] = "projection_y_coordinate"
        ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
        ds.y.attrs["units"] = "m"

        ds.x.attrs["axis"] = "X"
        ds.x.attrs["standard_name"] = "projection_x_coordinate"
        ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
        ds.x.attrs["units"] = "m"

    return ds


def group_netcdf_by_date(
    files: Iterable[Filename],
    file_date_fmt: str = 'datetime64[h]',
) -> dict[datetime.datetime, list[Filename]]:
    """Combine files by date and time into a dict.

    Parameters
    ----------
    files : Iterable[Filename]
        Path to folder containing files with dates in the time dataset.
    file_date_fmt : str
        Format of the date in the dataset.
       

    Returns
    -------
    dict
        key is a datetime in the time dataset of netcdf file.
        Value is a list of Paths on that date.
        E.g.:
        {(datetime.datetime(2017, 10, 13, 18),
          [Path(...)
            Path(...),
            ...]),
         (datetime.datetime(2017, 10, 25, 18),
          [Path(...)
            Path(...),
            ...]),
        }
    """
    # collapse into groups of dates
    # Use a `defaultdict` so we dont have to sort the files by date in advance,
    # but rather just extend the list each time there's a new group
    grouped_images: dict[datetime.datetime, list[Filename]] = defaultdict(
        list
    )
    def get_dates_netcdf(filename: Filename):
        ds = xarray.open_mfdataset(filename)
        date_time = ds.time.values.astype(file_date_fmt).astype(datetime.datetime)
        return date_time
        
    for date, g in itertools.groupby(
        files, key=lambda x: tuple(get_dates_netcdf(x))
    ):
        grouped_images[date[0]].extend(list(g))
    return grouped_images


def _convert_NetCDF(weather_model_file,
                    lat_lon_bounds,
                    weather_model_output_dir,
                    weather_model: TropoModel):
    '''
    Convert the weather model NetCDF to working NetCDF
    Parameters
    ----------
        weather_model_file: str
        The ECMWF/HRES NetCDF weather model file
        lat_lon_bounds: list
            bounding box 
        weather_model_output_dir: str
            the output directory of the internal NetCDF file
    Returns
    -------
            the path of the output NetCDF file
        '''
    
    if weather_model == TropoModel.HRES:
        from dolphin.atmosphere.weatherModel import HRES 
        wmodel = HRES()
    elif weather_model == TropoModel.ECMWF:
        from dolphin.atmosphere.weatherModel import ECMWF
        wmodel = ECMWF()
    else:
        FileExistsError("Weather model not supported")

    os.makedirs(weather_model_output_dir, exist_ok=True)
    ds = xarray.open_dataset(weather_model_file)

    # Get the datetime of the weather model file
    weather_model_time = ds.time.values.astype('datetime64[h]').astype(datetime.datetime)[0]
    
    # Set up the time, Lat/Lon, and working location, where
    # the lat/lon bounds are applied to clip the global
    # weather model to minimize the data processing
    wmodel.setTime(weather_model_time)
    wmodel.set_latlon_bounds(ll_bounds = lat_lon_bounds)
    wmodel.set_wmLoc(weather_model_output_dir)

    # Load the ECMWF NetCDF weather model
    wmodel.load_weather(weather_model_file)

    # Process the weather model data
    wmodel._find_e()
    wmodel._uniform_in_z(_zlevels=None)

    # This function fills the NaNs with 0
    wmodel._checkForNans()

    wmodel._get_wet_refractivity()
    wmodel._get_hydro_refractivity()
    wmodel._adjust_grid(wmodel.get_latlon_bounds())

    # Compute Zenith delays at the weather model grid nodes
    wmodel._getZTD()

    output_file = wmodel.out_file(weather_model_output_dir)
    wmodel._out_name =  output_file

    # Return the ouput file if it exists
    if os.path.exists(output_file):
        return output_file
    else:
        # Write to hard drive
        return wmodel.write()