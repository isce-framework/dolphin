import numpy as np

g0 = np.float64(9.80665)  # Standard gravitational constant
G1 = np.float64(
    9.80616
)  # Gravitational constant @ 45° latitude used for corrections of
# earth's centrifugal force
R_EARTH_MAX_WGS84 = Rmax = 6378137
R_EARTH_MIN_WGS84 = Rmin = 6356752


def find_svp(t):
    """Calculate standard vapor presure. Should be model-specific."""
    # From TRAIN:
    # Could not find the wrf used equation as they appear to be
    # mixed with latent heat etc. Istead I used the equations used
    # in ERA-I (see IFS documentation part 2: Data assimilation
    # (CY25R1)). Calculate saturated water vapour pressure (svp) for
    # water (svpw) using Buck 1881 and for ice (swpi) from Alduchow
    # and Eskridge (1996) euation AERKi

    # TODO: figure out the sources of all these magic numbers and move
    # them somewhere more visible.
    # TODO: (Jeremy) - Need to fix/get the equation for the other
    # weather model types. Right now this will be used for all models,
    # except WRF, which is yet to be implemented in my new structure.
    t1 = 273.15  # O Celsius
    t2 = 250.15  # -23 Celsius

    tref = t - t1
    wgt = (t - t2) / (t1 - t2)
    svpw = 6.1121 * np.exp((17.502 * tref) / (240.97 + tref))
    svpi = 6.1121 * np.exp((22.587 * tref) / (273.86 + tref))

    svp = svpi + (svpw - svpi) * wgt**2
    ix_bound1 = t > t1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = t < t2
    svp[ix_bound2] = svpi[ix_bound2]

    svp = svp * 100
    return svp.astype(np.float32)


# def get_mapping(proj):
#     '''Get CF-complient projection information from a proj'''
#     # In case of WGS-84 lat/lon, keep it simple
#     if proj.to_epsg()==4326:
#         return 'WGS84'
#     else:
#         return proj.to_wkt()


# def checkContainment_raw(path_wm_raw,
#                         ll_bounds,
#                         buffer_deg: float = 1e-5) -> bool:
#     """"
#     Checks if existing raw weather model contains
#     requested ll_bounds

#     Args:
#     ----------
#     path_wm_raw : path to downloaded, uncropped weather model file
#     ll_bounds: an array of floats (SNWE) demarcating bbox of targets
#     buffer_deg : float
#         For x-translates for extents that lie outside of world bounding box,
#         this ensures that translates have some overlap. The default is 1e-5
#         or ~11.1 meters.

#     Returns:
#     -------
#     bool
#         True if weather model contains bounding box of OutLats and outLons
#         and False otherwise.
#     """
#     import xarray as xr
#     ymin_input, ymax_input, xmin_input, xmax_input = ll_bounds
#     input_box   = box(xmin_input, ymin_input, xmax_input, ymax_input)

#     with xr.open_dataset(path_wm_raw) as ds:
#         try:
#             ymin, ymax = ds.latitude.min(), ds.latitude.max()
#             xmin, xmax = ds.longitude.min(), ds.longitude.max()
#         except:
#             ymin, ymax = ds.y.min(), ds.y.max()
#             xmin, xmax = ds.x.min(), ds.x.max()

#         xmin, xmax = np.mod(np.array([xmin, xmax])+180, 360) - 180
#         weather_model_box = box(xmin, ymin, xmax, ymax)

#     world_box  = box(-180, -90, 180, 90)

#     # Logger
#     input_box_str = [f'{x:1.2f}' for x in [xmin_input, ymin_input,
#                                             xmax_input, ymax_input]]
#     weath_box_str = [f'{x:1.2f}' for x in [xmin, ymin, xmax, ymax]]

#     weath_box_str = ', '.join(weath_box_str)
#     input_box_str = ', '.join(input_box_str)


#     # If the bounding box goes beyond the normal world extents
#     # Look at two x-translates, buffer them, and take their union.
#     if not world_box.contains(weather_model_box):
#         logger.info('Considering x-translates of weather model +/-360 '
#                     'as bounding box outside of -180, -90, 180, 90')
#         translates = [weather_model_box.buffer(buffer_deg),
#                         translate(weather_model_box,
#                                 xoff=360).buffer(buffer_deg),
#                         translate(weather_model_box,
#                                 xoff=-360).buffer(buffer_deg)
#                         ]
#         weather_model_box = unary_union(translates)

#     return weather_model_box.contains(input_box)


def calcgeoh(
    lnsp: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    R_d: int,
    num_levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate pressure, geopotential, and geopotential height.

    from the surface pressure and model levels provided by a weather model.
    The model levels are numbered from the highest elevation to the lowest.

    Parameters
    ----------
    lnsp: ndarray
        array of log surface pressure
    t: ndarray
        cube of temperatures
    q: ndarray
        cube of specific humidity
    z: ndarray
        cube of surface heights
    a: ndarray
        vector of a values
    b: ndarray
        vector of b values
    R_d: int
        model parameter
    num_levels: int
        integer number of model levels

    Returns
    -------
    geopotential : np.ndarray
        The geopotential in units of height times acceleration
    pressurelvs : np.ndarray
        The pressure at each of the model levels for each of
                    the input points
    geoheight : np.ndarray
        The geopotential heights

    """
    geopotential = np.zeros_like(t)
    pressurelvs = np.zeros_like(geopotential)
    geoheight = np.zeros_like(geopotential)

    # log surface pressure
    # Note that we integrate from the ground up, so from the largest model level to 0
    sp = np.exp(lnsp)

    if len(a) != num_levels + 1 or len(b) != num_levels + 1:
        raise ValueError(
            "I have here a model with {} levels, but parameters a ".format(num_levels)
            + "and b have lengths {} and {} respectively. Of ".format(len(a), len(b))
            + "course, these three numbers should be equal."
        )

    # Integrate up into the atmosphere from *lowest level*
    z_h = 0  # initial value
    for lev, t_level, q_level in zip(range(num_levels, 0, -1), t[::-1], q[::-1]):

        # lev is the level number 1-60, we need a corresponding index
        # into ts and qs
        # ilevel = num_levels - lev # << this was Ray's original, but is a typo
        # because indexing like that results in pressure and height arrays that
        # are in the opposite orientation to the t/q arrays.
        ilevel = lev - 1

        # compute moist temperature
        t_lev = t_level * (1 + 0.609133 * q_level)

        # compute the pressures (on half-levels)
        Ph_lev = a[lev - 1] + (b[lev - 1] * sp)
        Ph_levplusone = a[lev] + (b[lev] * sp)

        pressurelvs[ilevel] = (
            Ph_lev  # + Ph_levplusone) / 2
            # average pressure at half-levels above and below
        )

        if lev == 1:
            dlogP = np.log(Ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(Ph_levplusone) - np.log(Ph_lev)
            alpha = 1 - ((Ph_lev / (Ph_levplusone - Ph_lev)) * dlogP)

        TRd = t_lev * R_d

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f = z_h + TRd * alpha + z

        # Geopotential (add in surface geopotential)
        geopotential[ilevel] = z_f
        geoheight[ilevel] = geopotential[ilevel] / g0

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h += TRd * dlogP

    return geopotential, pressurelvs, geoheight


def geo_to_ht(lats: np.ndarray, hts: np.ndarray) -> np.ndarray:
    """Convert geopotential height to ellipsoidal heights.

    referenced to WGS84.
    Note that this formula technically computes height above
    geoid (geometric height) but the geoid is actually a
    perfect sphere; Thus returned heights are above a reference
    ellipsoid, which most assume to be a sphere.
    However, by calculating the ellipsoid here we directly
    reference to WGS84.

    # h = (geopotential * Re) / (g0 * Re - geopotential)
    # Assumes a sphere instead of an ellipsoid

    Parameters
    ----------
    lats : np.ndarray
        latitude of points of interest
    hts : np.ndarray
        geopotential height at points of interest

    Returns
    -------
    ndarray: geometric heights.
        These are approximate ellipsoidal heights referenced to WGS84

    """
    g_ll = _get_g_ll(lats)  # gravity function of latitude
    Re = get_earth_radius(lats)  # Earth radius function of latitude

    # Calculate Geometric Height, h
    h = (hts * Re) / (g_ll / g0 * Re - hts)

    return h


def _get_g_ll(lats):
    """Compute the variation in gravity constant with latitude."""
    return G1 * (1 - 0.002637 * cosd(2 * lats) + 0.0000059 * (cosd(2 * lats)) ** 2)


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def get_earth_radius(lats: np.ndarray) -> np.ndarray:
    """Get the earth radius as a function of latitude for WGS84.

    Parameters
    ----------
    lats : np.ndarray
        ndarray of geodetic latitudes in degrees

    Returns
    -------
    np.ndarray
        ndarray of earth radius at each latitude

    Example
    -------
    >>> import numpy as np
    >>> output = get_earth_radius(np.array([0, 30, 45, 60, 90]))
    >>> output
    array([6378137. , 6372770.5219805 , 6367417.56705189, 6362078.07851428, 6356752. ])
    >>> assert output[0] == 6378137  # (Rmax)
    >>> assert output[-1] == 6356752  # (Rmin)

    """
    return np.sqrt(1 / (((cosd(lats) ** 2) / Rmax**2) + ((sind(lats) ** 2) / Rmin**2)))


def sind(x):
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def pad_lower(invar):
    """Add a layer of data below the lowest current z-level at height zmin."""
    new_var = _least_nonzero(invar)
    return np.concatenate((new_var[:, :, np.newaxis], invar), axis=2)


def _least_nonzero(a):
    """Fill in a flat array with the first non-nan value in the last dimension.

    Useful for interpolation below the bottom of the weather model.
    """
    mgrid_index = tuple(slice(None, d) for d in a.shape[:-1])
    return a[(*tuple(np.mgrid[mgrid_index]), (~np.isnan(a)).argmax(-1))]
