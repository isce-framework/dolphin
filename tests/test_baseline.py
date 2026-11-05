from pathlib import Path

import numpy as np
import pytest

from dolphin import baseline

isce3 = pytest.importorskip("isce3")


def _geometry():
    """Target on the ellipsoid, a right-looking reference satellite and its look."""
    ell = isce3.core.Ellipsoid()
    llh = np.deg2rad([-120.0, 35.0, 0.0]).reshape((3, 1))
    target = np.ravel(ell.lon_lat_to_xyz(llh))
    up = target / np.linalg.norm(target)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    ref_pos = target + 700e3 * up + 400e3 * east
    look = (target - ref_pos) / np.linalg.norm(target - ref_pos)
    across = np.cross(look, north)
    return ell, llh, ref_pos, 7500.0 * north, look, across / np.linalg.norm(across)


def _ranges(llh, ell, ref_pos, sec_pos):
    target = np.ravel(ell.lon_lat_to_xyz(llh))
    return np.linalg.norm(target - ref_pos), np.linalg.norm(target - sec_pos)


@pytest.fixture
def cslc_files():
    p = Path(__file__).parent / "data/baseline_cslcs"
    return sorted(p.glob("*.h5"))


def test_baseline_compute(cslc_files):
    ref_file, sec_file = cslc_files
    lon, lat, baseline_grid = baseline.compute_baselines(
        h5file_ref=ref_file, h5file_sec=sec_file
    )
    assert lon.shape == lat.shape == baseline_grid.shape == (46, 199)
    assert np.isfinite(baseline_grid).all()
    pytest.approx([baseline_grid.min(), 134.98131295549837])
    pytest.approx([baseline_grid.max(), 141.95610459910463])


@pytest.mark.parametrize("b_par", [100.0, 1000.0])
@pytest.mark.parametrize("b_perp", [10.0, 1.0, 0.1, 0.01, 0.0])
def test_compute_near_zero_perpendicular_baseline(b_par, b_perp):
    # Nearly parallel to the look, where the law of cosines gave NaN or a bias
    ell, llh, ref_pos, ref_vel, look, across = _geometry()
    sec_pos = ref_pos + b_par * look + b_perp * across
    ranges = _ranges(llh, ell, ref_pos, sec_pos)
    perp = baseline.compute(llh, ref_pos, sec_pos, *ranges, ref_vel, ell)
    assert np.isfinite(perp)
    assert abs(perp) == pytest.approx(b_perp, abs=1e-6)


def test_compute_sign_follows_side():
    ell, llh, ref_pos, ref_vel, _look, across = _geometry()
    plus_pos, minus_pos = ref_pos + 5.0 * across, ref_pos - 5.0 * across
    plus = baseline.compute(
        llh, ref_pos, plus_pos, *_ranges(llh, ell, ref_pos, plus_pos), ref_vel, ell
    )
    minus = baseline.compute(
        llh, ref_pos, minus_pos, *_ranges(llh, ell, ref_pos, minus_pos), ref_vel, ell
    )
    assert plus == pytest.approx(-minus)
    assert abs(plus) == pytest.approx(5.0)
