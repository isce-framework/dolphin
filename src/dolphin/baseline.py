import isce3
import numpy as np
from numpy.typing import ArrayLike
from opera_utils import (
    get_cslc_orbit,
    get_lonlat_grid,
    get_radar_wavelength,
)

from dolphin._types import Filename


def compute(
    llh: ArrayLike,
    ref_pos: ArrayLike,
    sec_pos: ArrayLike,
    ref_range: float,
    sec_range: float,
    ref_vel: ArrayLike,
    ell: isce3.core.Ellipsoid,
):
    """Compute the perpendicular baseline at a given geographic position.

    Parameters
    ----------
    llh : ArrayLike
        Lon/Lat/Height vector specifying the target position.
        Lon and Lat must be in radians, not degrees.
    ref_pos : ArrayLike
        Reference position vector (x, y, z) in ECEF coordinates.
    sec_pos : ArrayLike
        Secondary position vector (x, y, z) in ECEF coordinates.
    ref_range : float
        Range from the reference satellite to the target.
    sec_range : float
        Range from the secondary satellite to the target.
    ref_vel : ArrayLike
        Velocity vector (vx, vy, vz) of the reference satellite in ECEF coordinates.
    ell : isce3.core.Ellipsoid
        Ellipsoid for the target surface.

    Returns
    -------
    float
        Perpendicular baseline, in meters.

    """
    # Difference in position between the two passes
    baseline = np.linalg.norm(sec_pos - ref_pos)

    # Compute angle between LOS vector and baseline vector
    # via the law of cosines
    cos_theta = (ref_range**2 + baseline**2 - sec_range**2) / (2 * ref_range * baseline)

    sin_theta = np.sqrt(1 - cos_theta**2)
    perp = baseline * sin_theta
    # parallel_baseline = baseline * cosine_theta

    target_xyz = ell.lon_lat_to_xyz(llh)
    direction = np.sign(
        np.dot(np.cross(target_xyz - ref_pos, sec_pos - ref_pos), ref_vel)
    )

    return direction * perp


def compute_baselines(
    h5file_ref: Filename,
    h5file_sec: Filename,
    height: float = 0.0,
    latlon_subsample: int = 100,
    threshold: float = 1e-08,
    maxiter: int = 50,
    delta_range: float = 10.0,
):
    """Compute the perpendicular baseline at a subsampled grid for two CSLCs.

    Parameters.
    ----------
    h5file_ref : Filename
        Path to reference OPERA S1 CSLC HDF5 file.
    h5file_sec : Filename
        Path to secondary OPERA S1 CSLC HDF5 file.
    height: float
        Target height to use for baseline computation.
        Default = 0.0
    latlon_subsample: int
        Factor by which to subsample the CSLC latitude/longitude grids.
        Default = 30
    threshold : float
        isce3 geo2rdr: azimuth time convergence threshold in meters
        Default = 1e-8
    maxiter : int
        isce3 geo2rdr: Maximum number of Newton-Raphson iterations
        Default = 50
    delta_range : float
        isce3 geo2rdr: Step size used for computing derivative of doppler
        Default = 10.0

    Returns
    -------
    lon : np.ndarray
        2D array of longitude coordinates in degrees.
    lat : np.ndarray
        2D array of latitude coordinates in degrees.
    baselines : np.ndarray
        2D array of perpendicular baselines

    """
    lon_grid, lat_grid = get_lonlat_grid(h5file_ref, subsample=latlon_subsample)
    lon_arr = lon_grid.ravel()
    lat_arr = lat_grid.ravel()

    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()
    wavelength = get_radar_wavelength(h5file_ref)
    side = isce3.core.LookSide.Right

    orbit_ref = get_cslc_orbit(h5file_ref)
    orbit_sec = get_cslc_orbit(h5file_sec)

    baselines = []
    for lon, lat in zip(lon_arr, lat_arr, strict=False):
        llh_rad = np.deg2rad([lon, lat, height]).reshape((3, 1))
        az_time_ref, range_ref = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_ref,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )
        az_time_sec, range_sec = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_sec,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )

        pos_ref, velocity = orbit_ref.interpolate(az_time_ref)
        pos_sec, _ = orbit_sec.interpolate(az_time_sec)
        b = compute(
            llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
        )

        baselines.append(b)

    baseline_grid = np.array(baselines).reshape(lon_grid.shape)
    return lon_grid, lat_grid, baseline_grid
