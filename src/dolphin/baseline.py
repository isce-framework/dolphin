import isce3
import numpy as np


def compute(
    llh: ArrayLike,
    ref_pos: ArrayLike,
    sec_pos: ArrayLike,
    ref_rng: float,
    sec_rng: float,
    ref_vel: ArrayLike,
    ell: isce3.core.Ellipsoid,
):
    """Compute the perpendicular baseline at a given geographic position.

    Parameters
    ----------
    llh : ArrayLike
        Lon/Lat/Height vector specifying the target position.
    ref_pos : ArrayLike
        Reference position vector (x, y, z) in ECEF coordinates.
    sec_pos : ArrayLike
        Secondary position vector (x, y, z) in ECEF coordinates.
    ref_rng : float
        Range from the reference satellite to the target.
    sec_rng : float
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
    costheta = (ref_rng**2 + baseline**2 - sec_rng**2) / (2 * ref_rng * baseline)

    sintheta = np.sqrt(1 - costheta**2)
    perp = baseline * sintheta
    # par = baseline * costheta

    targ_xyz = ell.lon_lat_to_xyz(llh)
    direction = np.sign(
        np.dot(np.cross(targ_xyz - ref_pos, sec_pos - ref_pos), ref_vel)
    )

    return direction * perp
