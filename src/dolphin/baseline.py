import isce3
import numpy


def perpendicular_baseline(
    llh,
    ell: isce3.core.Ellipsoid,
    dop: isce3.core.LUT2d,
    side: isce3.core.LookSide,
    wvl: float,  # wavelength
    ref_orbit: isce3.core.Orbit,
    sec_orbit: isce3.core.Orbit,
):
    """Compute perpendicular baseline at a given geometric position.

    Parameters
    ----------
    llh :
        Lon/Lat/Height vector specifying target position
    ell : isce3.core.Ellipsoid
        Ellipsoid for the target surface
    dop : isce3.core.LUT2d
        Doppler LUT
    side : isce3.core.LookSide
        Look side
    wvl : float
        Wavelength
    ref_orbit : isce3.core.Orbit
        Reference orbit
    sec_orbit : isce3.core.Orbit
        Secondary orbit

    Returns
    -------
    float
        Perpendicular baseline, in meters

    """
    az, ref_rng = isce3.geometry.geo2rdr(llh, ell, ref_orbit, dop, wvl, side)
    ref_pos, vel = ref_orbit.interpolate(az)

    az, sec_rng = isce3.geometry.geo2rdr(llh, ell, sec_orbit, dop, wvl, side)
    sec_pos, _ = sec_orbit.interpolate(az)

    # Difference in position between the two passes
    baseline = numpy.linalg.norm(sec_pos - ref_pos)

    # Compute angle between LOS vector and baseline vector
    # via the law of cosines
    costheta = (ref_rng**2 + baseline**2 - sec_rng**2) / (2 * ref_rng * baseline)

    sintheta = numpy.sqrt(1 - costheta**2)
    perp = baseline * sintheta
    # par = baseline * costheta

    targ_xyz = ell.lon_lat_to_xyz(llh)
    direction = numpy.sign(
        numpy.dot(numpy.cross(targ_xyz - ref_pos, sec_pos - ref_pos), vel)
    )

    return direction * perp
