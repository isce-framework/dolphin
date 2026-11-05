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
    ref_range: float,  # noqa: ARG001  kept for callers, see docstring
    sec_range: float,  # noqa: ARG001
    ref_vel: ArrayLike,
    ell: isce3.core.Ellipsoid,
):
    """Compute the perpendicular baseline at a given geographic position.

    The baseline component perpendicular to the reference line of sight,
    |B x l| for the unit look vector l, so it stays accurate as it goes to zero.

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
        Range from the reference satellite to the target. Unused, kept so
        existing callers (disp-s1) keep working.
    sec_range : float
        Range from the secondary satellite to the target. Unused, as above.
    ref_vel : ArrayLike
        Velocity vector (vx, vy, vz) of the reference satellite in ECEF coordinates.
    ell : isce3.core.Ellipsoid
        Ellipsoid for the target surface.

    Returns
    -------
    float
        Perpendicular baseline, in meters.

    """
    target_xyz = np.ravel(ell.lon_lat_to_xyz(llh))
    baseline_vec = np.ravel(sec_pos) - np.ravel(ref_pos)

    # Unit look vector, reference satellite to target
    look = target_xyz - np.ravel(ref_pos)
    look /= np.linalg.norm(look)

    # Not the law of cosines on the two ranges: it cancels as B_perp -> 0, gives
    # |cos| > 1 (NaN) there, and is biased when sec_pos no longer matches its range
    perp = np.linalg.norm(np.cross(baseline_vec, look))

    direction = np.sign(np.dot(np.cross(look, baseline_vec), np.ravel(ref_vel)))

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

        # Accounting for along-track shift
        rvelunit = velocity / np.linalg.norm(velocity)
        pos_sec = pos_sec - np.dot(pos_sec - pos_ref, rvelunit) * rvelunit

        b = compute(
            llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
        )

        baselines.append(b)

    baseline_grid = np.array(baselines).reshape(lon_grid.shape)
    return lon_grid, lat_grid, baseline_grid
