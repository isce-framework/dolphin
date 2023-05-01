import numpy as np
import pytest

from dolphin import shp
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.fixture(scope="module")
def slcs(shape=(30, 11, 11)):
    return 20 * (np.random.rand(*shape) + 1j * np.random.rand(*shape))


def test_shp_tf(slcs):
    amp = np.abs(slcs)
    mean = np.mean(amp, axis=0)
    var = np.var(amp, axis=0)
    # Looking at the entire stack
    half_rowcol = (5, 5)
    n = slcs.shape[0]

    # First try a tiny alpha
    neighbors = shp.estimate_neighbors_tf(mean, var, half_rowcol, n, alpha=1e-6)
    shps_center = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_center.all()

    # Check the edges are cut off
    top_left = neighbors[0, 0]
    assert top_left[:5, :5].all()
    assert top_left[6:, :].sum() == top_left[:, 6:].sum() == 0

    # Next try a large alpha (should be no neighbors)
    neighbors = shp.estimate_neighbors_tf(mean, var, half_rowcol, n, alpha=1.0)
    shps_center = neighbors[5, 5]
    assert shps_center.sum() == 1  # only itself
    assert shps_center[5, 5] == 1


def test_shp_tf_half_mean_different(slcs):
    """Run a test where half the image has different mean"""
    amp = np.abs(slcs)
    mean = np.mean(amp, axis=0)
    var = np.var(amp, axis=0)
    n = slcs.shape[0]

    half_rowcol = (5, 5)
    # make the top half different amplitude
    mean2 = mean.copy()
    mean2[:5, :] += 500
    # For this test, make all variances equal (just checking mean)
    var[:] = var[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors_tf(mean2, var, half_rowcol, n, alpha=0.01)
    shps_center = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_center[5:, :].all()
    assert not shps_center[:5, :].any()


def test_shp_tf_half_var_different(slcs):
    """Run a test where half the image has different variance"""
    amp = np.abs(slcs)
    mean = np.mean(amp, axis=0)
    var = np.var(amp, axis=0)
    n = slcs.shape[0]

    half_rowcol = (5, 5)
    # make the top half different amplitude
    var2 = var.copy()
    var2[:5, :] += 1000
    # For this test, make all means equal (just checking var)
    mean[:] = mean[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors_tf(mean, var2, half_rowcol, n, alpha=0.01)
    shps_center = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_center[5:, :].all()
    assert not shps_center[:5, :].any()


@pytest.mark.parametrize("alpha", [0.01, 0.05])
@pytest.mark.parametrize("strides", [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
def test_shp_tf_statistics(alpha, strides):
    """Check that with repeated tries, the alpha is correct."""

    nsim = 500
    shape = (30, 11, 11)
    half_rowcol = (5, 5)
    frac_shps = np.zeros(nsim)
    slc_rows = np.random.rand(nsim, *shape) + 1j * np.random.rand(nsim, *shape)
    for i in range(nsim):
        slcs = slc_rows[i]
        amp = np.abs(slcs)
        mean = np.mean(amp, axis=0)
        var = np.var(amp, axis=0)
        n = slcs.shape[0]

        neighbors = shp.estimate_neighbors_tf(
            mean, var, half_rowcol, n, strides=strides, alpha=alpha
        )

        # neighbors = shp.estimate_neighbors_ks(
        #     np.abs(slcs), half_rowcol, strides=strides, alpha=alpha, is_sorted=False
        # )
        # shps_center = neighbors_ks[5, 5]
        out_rows, out_cols = neighbors.shape[:2]
        shps_center = neighbors[out_rows // 2, out_cols // 2]
        frac_shps[i] = shps_center.sum() / (shps_center.size - 1)  # don't count center

    # Check that the mean number of SHPs is close to 5%
    tol_pct = 3
    assert 100 * np.abs((frac_shps.mean() - (1 - alpha))) < tol_pct
