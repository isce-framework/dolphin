import numpy as np
import pytest
from scipy.stats import rayleigh

from dolphin import shp
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.fixture(scope="module")
def slcs(shape=(30, 11, 11)):
    return 20 * (np.random.rand(*shape) + 1j * np.random.rand(*shape))


# the TF and GLRT methods have same API for estimate_neighbors
@pytest.mark.parametrize("method", ["tf", "glrt"])
def test_shp_glrt_tf(slcs, method):
    amp_stack = np.abs(slcs)
    mean = np.mean(amp_stack, axis=0)
    var = np.var(amp_stack, axis=0)

    halfwin_rowcol = (5, 5)  # Looking at the entire stack
    nslc = slcs.shape[0]

    # First try a tiny alpha
    neighbors = shp.estimate_neighbors(
        mean=mean,
        var=var,
        halfwin_rowcol=halfwin_rowcol,
        nslc=nslc,
        alpha=1e-6,
        method=method,
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_mid_pixel.all()

    # Check the edges are cut off
    top_left = neighbors[0, 0]
    assert top_left[:5, :5].all()
    assert top_left[6:, :].sum() == top_left[:, 6:].sum() == 0

    # Next try a large alpha (should be no neighbors)
    neighbors = shp.estimate_neighbors(
        mean=mean,
        var=var,
        halfwin_rowcol=halfwin_rowcol,
        nslc=nslc,
        alpha=1.0,
        method="tf",
    )
    shps_mid_pixel = neighbors[5, 5]
    assert shps_mid_pixel.sum() == 1  # only itself
    assert shps_mid_pixel[5, 5] == 1


def test_shp_ks(slcs):
    amp_stack = np.abs(slcs)
    halfwin_rowcol = (5, 5)  # Looking at the entire stack
    nslc = slcs.shape[0]

    # First try a tiny alpha
    neighbors = shp.estimate_neighbors(
        amp_stack=amp_stack,
        halfwin_rowcol=halfwin_rowcol,
        nslc=nslc,
        alpha=1e-6,
        method="ks",
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_mid_pixel.all()

    # Check the edges are cut off and all zeros
    # NOTE: this is different than TF test since KS is slower
    # TODO if i want them to be exactly the same...
    top_left = neighbors[0, 0]
    assert not top_left[:5, :5].any()

    # Next try a large alpha (should be no neighbors)
    neighbors = shp.estimate_neighbors(
        amp_stack=amp_stack,
        halfwin_rowcol=halfwin_rowcol,
        nslc=nslc,
        alpha=1.0,
        method="ks",
    )
    shps_mid_pixel = neighbors[5, 5]
    assert shps_mid_pixel.sum() == 1  # only itself
    assert shps_mid_pixel[5, 5] == 1


def test_shp_tf_half_mean_different(slcs):
    """Run a test where half the image has different mean"""
    amp_stack = np.abs(slcs)
    mean = np.mean(amp_stack, axis=0)
    var = np.var(amp_stack, axis=0)
    nslc = slcs.shape[0]

    halfwin_rowcol = (5, 5)
    # make the top half different amplitude
    mean2 = mean.copy()
    mean2[:5, :] += 500
    # For this test, make all variances equal (just checking mean)
    var[:] = var[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors(
        mean=mean2, var=var, halfwin_rowcol=halfwin_rowcol, nslc=nslc, alpha=0.01
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_mid_pixel[5:, :].all()
    assert not shps_mid_pixel[:5, :].any()


def test_shp_tf_half_var_different(slcs):
    """Run a test where half the image has different variance"""
    amp_stack = np.abs(slcs)
    mean = np.mean(amp_stack, axis=0)
    var = np.var(amp_stack, axis=0)
    nslc = slcs.shape[0]

    halfwin_rowcol = (5, 5)
    # make the top half different amplitude
    var2 = var.copy()
    var2[:5, :] += 1000
    # For this test, make all means equal (just checking var)
    mean[:] = mean[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors(
        mean=mean, var=var2, halfwin_rowcol=halfwin_rowcol, nslc=nslc, alpha=0.01
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_mid_pixel[5:, :].all()
    assert not shps_mid_pixel[:5, :].any()


@pytest.mark.parametrize("method", ["tf", "glrt"])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
@pytest.mark.parametrize("strides", [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
def test_shp_tf_statistics(method, alpha, strides):
    """Check that with repeated tries, the alpha is correct."""

    nsim = 500
    shape = (30, 11, 11)
    halfwin_rowcol = (5, 5)
    shp_counts = np.zeros(nsim)
    # slc_rows = np.random.rand(nsim, *shape) + 1j * np.random.rand(nsim, *shape)

    slc_rows = rayleigh.rvs(scale=10, size=(nsim, *shape))
    for i in range(nsim):
        slcs = slc_rows[i]
        amp_stack = np.abs(slcs)
        mean = np.mean(amp_stack, axis=0)
        var = np.var(amp_stack, axis=0)
        nslc = slcs.shape[0]

        neighbors = shp.estimate_neighbors(
            mean=mean,
            var=var,
            halfwin_rowcol=halfwin_rowcol,
            nslc=nslc,
            strides=strides,
            alpha=alpha,
            method=method,
        )

        out_rows, out_cols = neighbors.shape[:2]
        shps_mid_pixel = neighbors[out_rows // 2, out_cols // 2]
        # dont count center (self) as one
        shp_counts[i] = shps_mid_pixel.sum() - 1

    # Check that the mean number of SHPs is close to 5%
    nbox = shape[1] * shape[2] - 1
    shp_frac = shp_counts.mean() / nbox
    tol_pct = 3
    assert 100 * np.abs((shp_frac - (1 - alpha))) < tol_pct


# def test_glrt_same_distribution():
#     nslc = 100
#     num_tests = 500
#     alpha = 0.05
#     data = rayleigh.rvs(scale=1, size=(nslc, num_tests))

#     mean= np.mean(data, axis=0)
#     var= np.var(data, axis=0)
#     half_window = (0, 0)

#     neighbors = shp.estimate_neighbors(
#             mean=mean,
#             var=var,
#             halfwin_rowcol=halfwin_rowcol,
#             # amp_stack=amp_stack,
#             nslc=nslc,
#             strides=strides,
#             alpha=alpha,
#             method=method,
#         )

#     rejected = np.sum(~neighbors[5, 5]) - 1  # skip counting the self pixel

#     assert np.abs(rejected / num_tests - alpha) < 0.01
