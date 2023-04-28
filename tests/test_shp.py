import numpy as np
import pytest

from dolphin import shp
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.fixture(scope="module")
def slcs(shape=(30, 11, 11)):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


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

    # Check the edges are cut off (not even iterated over) and all 0
    assert neighbors[:5, :].sum() == 0

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
    mean2 = mean.copy()
    mean2[:5, :] += 200  # make the top half different amplitude

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
    var2 = var.copy()
    var2[:5, :] += 500  # make the top half different amplitude

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors_tf(mean, var2, half_rowcol, n, alpha=0.01)
    shps_center = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_center[5:, :].all()
    assert not shps_center[:5, :].any()


def test_shp_tf_statistics():
    """Check that with repeated tries, the alpha is correct."""

    nsim = 100
    shape = (30, 11, 11)
    num_shps = np.zeros(nsim)
    for i in range(nsim):
        slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        amp = np.abs(slcs)
        mean = np.mean(amp, axis=0)
        var = np.var(amp, axis=0)
        n = slcs.shape[0]

        half_rowcol = (5, 5)
        neighbors = shp.estimate_neighbors_tf(mean, var, half_rowcol, n, alpha=0.05)
        num_shps[i] = neighbors[5, 5].sum()

    # Check that the mean number of SHPs is close to 5%
    assert np.abs(100 * (num_shps.mean() / 11 * 11) - 5) < 1
