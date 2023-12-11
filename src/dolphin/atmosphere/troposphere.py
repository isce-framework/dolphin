from __future__ import annotations

import os
from os import fspath
from pathlib import Path
from typing import List, Optional
import datetime
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pyaps3 as pa
import opera_utils as oput
from dolphin._log import get_log
from dolphin._types import Bbox, Filename
from dolphin import io, stitching
logger = get_log(__name__)


###########

def estimate_tropospheric_delay(ifg_file_list: list[Path], slc_files: dict[datetime.date, list[Filename]],
                                troposphere_files: dict[datetime.date, list[Filename]],
                                geom_files: List[Path], dem_file: Optional[Path], output_dir: Path,
                                tropo_package: str, tropo_model: str, tropo_delay_type: str,
                                strides: dict[str, int] = {"x": 1, "y": 1}):
    
   """
    Estimate the tropospheric delay corrections for each interferogram.

    Parameters
    ----------
    ifg_file_list : List[Path]
        List of interferogram files.
    slc_files : Dict[datetime.date, List[Filename]]
        Dictionary of SLC files indexed by date.
    troposphere_files : Dict[datetime.date, List[Filename]]
        Dictionary of troposphere files indexed by date.
    geom_files : List[Path]
        List of geometry files.
    dem_file : Optional[Path]
        DEM file.
    output_dir : Path
        Output directory.
    tropo_package : str
        Troposphere processing package ('pyaps' or 'raider').
    tropo_model : str
        Tropospheric model (ERA5, HRES, ...). 
    tropo_delay_type : str
        Tropospheric delay type ('wet', 'hydrostatic', 'comb').
    strides : Dict[str, int], optional
        Strides for resampling, by default {"x": 1, "y": 1}.
    """

    # Read geogrid data
    xsize, ysize = io.get_raster_xysize(ifg_file_list[0])
    crs = io.get_raster_crs(ifg_file_list[0])
    gt = io.get_raster_gt(ifg_file_list[0])
    ycoord, xcoord = oput.create_yx_arrays(gt, (ysize, xsize)) # 500 m spacing
    epsg = crs.to_epsg()
    out_bounds = io.get_raster_bounds(ifg_file_list[0])
    
    # prepare geometry data
    print('\n'+'-'*80)
    print('Prepare geometry files...')
    geometry_dir = output_dir / "../../geometry"
    geometry_files = prepare_geometry(geometry_dir=geometry_dir,
                                      geo_files=geom_files, matching_file=ifg_file_list[0],
                                      dem_file=dem_file, epsg=epsg, out_bounds=out_bounds, 
                                      strides=strides)

    
    tropo_height_levels = np.concatenate(([-100], np.arange(0, 9000, 500)))

    # Note on inc_angle: This was a data cube, we are using a constant now and need to be updated 
    grid = {'xcoord': xcoord,
            'ycoord': ycoord,
            'height_levels': tropo_height_levels,
            'snwe': oput.get_snwe(epsg, out_bounds),
            'epsg': epsg,
            'geotransform': gt,
            'shape': (ysize, xsize),
            'crs': crs.to_wkt()} # cube

    
    if tropo_package.lower() == 'pyaps':
        tropo_run = compute_pyaps
    else:
        tropo_run = comput_raider

    tropo_delay_products = []
    # comb is short for the summation of wet and dry components
    if tropo_delay_type in ['wet', 'hydrostatic', 'comb']:
        if (tropo_delay_type == 'hydrostatic') and (tropo_package == 'raider'):
                delay_type = 'hydro'
        elif (tropo_delay_type == 'hydrostatic') and (tropo_package == 'pyaps'):
                delay_type = 'dry'
        else:
            delay_type = tropo_delay_type

        tropo_delay_products.append(delay_type)
    
    first_date = next(iter(slc_files))
    wavelength = oput.get_radar_wavelength(slc_files[first_date][0])
    for ifg in ifg_file_list:
        
        ref_date, sec_date = os.path.basename(fspath(ifg)).split('.')[0].split('_')
        for delayt in tropo_delay_products:
            tropo_delay_product_name = fspath(output_dir) + f'/{ref_date}_{sec_date}_tropoDelay_pyaps_{tropo_model}_LOS_{delayt}.tif'
            if os.path.exists(tropo_delay_product_name):
                run_or_skip = 'skip'
            else:
                run_or_skip = 'run'
        if run_or_skip == 'skip':
            logger.info(
                f"Tropospheric correction {os.path.basename(tropo_delay_product_name)} already exists, skipping"
            )
            continue

        reference_date = datetime.datetime.strptime(ref_date, '%Y%m%d').date()
        secondary_date = datetime.datetime.strptime(sec_date, '%Y%m%d').date()
        if reference_date in troposphere_files.keys()  and secondary_date in troposphere_files.keys():
            import pdb; pdb.set_trace()
            reference_time = oput.get_zero_doppler_time(slc_files[reference_date][0])
            secondary_time = oput.get_zero_doppler_time(slc_files[secondary_date][0])
            weather_model_params = {'reference_file':troposphere_files[reference_date],
                                    'secondary_file':troposphere_files[secondary_date],
                                    'output_dir': output_dir,
                                    'type':tropo_model.upper(),
                                    'reference_time': reference_time,
                                    'secondary_time': secondary_time,
                                    'wavelength':wavelength}
            troposphere_delay_datacube = tropo_run(tropo_delay_products=tropo_delay_products,
                                                   grid=grid, weather_model_params=weather_model_params)
            
            
            tropo_delay_2d = oput.compute_2d_delay(troposphere_delay_datacube, grid, geometry_files)
            write_tropo(tropo_delay_2d, ifg, output_dir)
      
        else:
            logger.warn(
                f"Weather-model files do not exist for interferogram {reference_date}-{secondary_date}, skipping"
            )
    
    return

def prepare_geometry(geometry_dir: Path,
                     geo_files: List[Path],
                     matching_file: Path,
                     dem_file: Optional[Path],
                     epsg: int, out_bounds: Bbox,
                     strides: dict[str, int] = {"x": 1, "y": 1}) ->dict[str, Path]: 
    """
    Prepare geometry files.

    Parameters
    ----------
    geometry_dir : Path
        Output directory for geometry files.
    geo_files : List[Path]
        List of geometry files.
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
    os.makedirs(geometry_dir, exist_ok=True)

    stitched_geo_list = {}

    # local_incidence_angle needed by anyone?
    datasets = ["los_east", "los_north"]
  
    for ds_name in datasets:
        outfile = geometry_dir / f"{ds_name}.tif"
        print(f"Creating {outfile}")
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
        stitched_geo_list['height'] = height_file
        if not os.path.exists(height_file):
            print(f"Creating {height_file}")
            stitching.warp_to_match(
                input_file=dem_file,
                match_file=matching_file,
                output_file=height_file,
                resample_alg="cubic",
            )

    return stitched_geo_list


def write_tropo(tropo_2d: dict, 
                inp_interferogram: Path, 
                out_dir: Path) -> None:
    
    """
    Write tropospheric delay data to files.

    Parameters
    ----------
    tropo_2d : Dict
        Dictionary containing tropospheric delay data.
    inp_interferogram : Path
        Input interferogram file.
    out_dir : Path
        Output directory.
    """
    dates = os.path.basename(inp_interferogram).split('.')[0]
    os.makedirs(out_dir, exist_ok=True)

    for key, value in tropo_2d.items():
        output = os.path.join(out_dir, f"{dates}_{key}.tif")
        io.write_arr(arr=value, output_name=output, like_filename=inp_interferogram) 
    return

def compute_pyaps(tropo_delay_products: List[str], 
                  grid: dict, weather_model_params: dict) -> dict:
    
    """
    Compute tropospheric delay datacube using PyAPS.

    Parameters
    ----------
    tropo_delay_products : List[str]
        List of tropospheric delay products.
    grid : Dict
        Dictionary containing grid information.
    weather_model_params : Dict
        Dictionary containing weather model parameters.

    Returns
    -------
    Dict
        Dictionary containing computed tropospheric delay datacube.
    """
    troposphere_delay_datacube = dict()
    
    # X and y for the entire datacube
    y_2d_radar, x_2d_radar = np.meshgrid(grid['ycoord'], grid['xcoord'], indexing='ij')

    # Lat/lon coordinates
    lat_datacube, lon_datacube = oput.transform_xy_to_latlon(grid['epsg'], x_2d_radar, y_2d_radar)

    for tropo_delay_product in tropo_delay_products:
        tropo_delay_datacube_list = []
        
        for index, hgt in enumerate(grid['height_levels']):
            dem_datacube = np.full(lat_datacube.shape, hgt)
            
            # Delay for the reference image
            ref_aps_estimator = pa.PyAPS(fspath(weather_model_params['reference_file'][0]),
                                         dem=dem_datacube,
                                         inc=0.0,
                                         lat=lat_datacube,
                                         lon=lon_datacube,
                                         grib=weather_model_params['type'],
                                         humidity='Q',
                                         model=weather_model_params['type'],
                                         verb=False,
                                         Del=tropo_delay_product)
            
            phs_ref = np.zeros((ref_aps_estimator.ny, ref_aps_estimator.nx), dtype=np.float32)
            ref_aps_estimator.getdelay(phs_ref)

            # Delay for the secondary image
            second_aps_estimator = pa.PyAPS(fspath(weather_model_params['secondary_file'][0]),
                                            dem=dem_datacube,
                                            inc=0.0,
                                            lat=lat_datacube,
                                            lon=lon_datacube,
                                            grib=weather_model_params['type'],
                                            humidity='Q',
                                            model=weather_model_params['type'],
                                            verb=False,
                                            Del=tropo_delay_product)

            phs_second = np.zeros((second_aps_estimator.ny, second_aps_estimator.nx), dtype=np.float32)
            second_aps_estimator.getdelay(phs_second)

            # Convert the delay in meters to radians
            tropo_delay_datacube_list.append(
                -(phs_ref - phs_second) * 4.0 * np.pi / float(weather_model_params['wavelength']))

            # Tropo delay datacube
        tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
        # Create a maksed datacube that excludes the NaN values
        tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

        # Save to the dictionary in memory
        model_type = weather_model_params['type']
        tropo_delay_product_name = f'tropoDelay_pyaps_{model_type}_Zenith_{tropo_delay_product}'
        troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube_masked


    return troposphere_delay_datacube

def comput_raider(tropo_delay_products: List[str], 
                  grid: dict, weather_model_params: dict) -> dict:
    
    """
    Compute tropospheric delay datacube using RAiDER.

    Parameters
    ----------
    tropo_delay_products : List[str]
        List of tropospheric delay products.
    grid : Dict
        Dictionary containing grid information.
    weather_model_params : Dict
        Dictionary containing weather model parameters.

    Returns
    -------
    Dict
        Dictionary containing computed tropospheric delay datacube.
    """
    from RAiDER.llreader import BoundingBox
    from RAiDER.losreader import Zenith
    from RAiDER.delay import tropo_delay as raider_tropo_delay


    reference_weather_model_file = weather_model_params['reference_file']
    secondary_weather_model_file = weather_model_params['secondary_file']
    
    troposphere_delay_datacube = dict()

    aoi = BoundingBox(grid['snwe'])
    aoi.xpts = grid['xcoord']
    aoi.ypts = grid['ycoord']

    # Zenith
    delay_direction_obj = Zenith()

    # Troposphere delay computation
    # Troposphere delay datacube computation
    tropo_delay_reference, _ = raider_tropo_delay(dt=fspath(weather_model_params['reference_time'][0]),
                                                  weather_model_file=reference_weather_model_file,
                                                  aoi=aoi,
                                                  los=delay_direction_obj,
                                                  height_levels=grid['height_levels'],
                                                  out_proj=grid['epsg'])
    
    tropo_delay_secondary, _ = raider_tropo_delay(dt=fspath(weather_model_params['secondary_time'][0]),
                                                  weather_model_file=secondary_weather_model_file,
                                                  aoi=aoi,
                                                  los=delay_direction_obj,
                                                  height_levels=grid['height_levels'],
                                                  out_proj=grid['epsg'])
    

    for tropo_delay_product in tropo_delay_products:

        # Compute troposphere delay with raider package
        # comb is the summation of wet and hydro components
        if tropo_delay_product == 'comb':
            tropo_delay = tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                    tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
        else:
            tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                    tropo_delay_secondary[tropo_delay_product]

        # Convert it to radians units
        tropo_delay_datacube = -tropo_delay * 4.0 * np.pi / weather_model_params['wavelength']

        # Create a maksed datacube that excludes the NaN values
        tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_datacube)

        # Interpolate to radar grid to keep its dimension consistent with other datacubes
        tropo_delay_interpolator = RegularGridInterpolator((tropo_delay_reference.z,
                                                            tropo_delay_reference.y,
                                                            tropo_delay_reference.x),
                                                           tropo_delay_datacube_masked,
                                                           method='linear', bounds_error=False)
        
        # Interpolate the troposphere delay
        hv, yv, xv = np.meshgrid(grid['height_levels'],
                                 grid['ycoord'],
                                 grid['xcoord'],
                                 indexing='ij')

        pnts = np.stack(
                (hv.flatten(), yv.flatten(), xv.flatten()), axis=-1)

        # Interpolate
        tropo_delay_datacube = tropo_delay_interpolator(pnts).reshape(hv.shape)


        # Save to the dictionary in memory
        model_type = weather_model_params['type']
        tropo_delay_product_name = f'tropoDelay_raider_{model_type}_Zenith_{tropo_delay_product}'
        troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube


    return troposphere_delay_datacube
