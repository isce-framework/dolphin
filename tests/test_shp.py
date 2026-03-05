import numpy as np
import pytest
from scipy.stats import rayleigh

from dolphin import shp
from dolphin.phase_link import simulate
from dolphin.shp import ShpMethod
from dolphin.shp._common import remove_unconnected

simulate._seed(1234)


@pytest.mark.parametrize("method", ["glrt", "ks"])
def test_shp_glrt_tf_smoketest(method):
    shape = (5, 50, 50)
    slcs = 20 * (np.random.rand(*shape) + 1j * np.random.rand(*shape))
    amp_stack = np.abs(slcs)
    mean = np.mean(amp_stack, axis=0)
    var = np.var(amp_stack, axis=0)

    # Make sure no errors
    shp.estimate_neighbors(
        mean=mean,
        var=var,
        halfwin_rowcol=(3, 5),
        nslc=slcs.shape[0],
        amp_stack=amp_stack,
        alpha=0.05,
        method=method,
    )


NUM_SLCS = 30


@pytest.fixture(scope="module")
def slcs(shape=(NUM_SLCS, 11, 11)):
    return 20 * (np.random.rand(*shape) + 1j * np.random.rand(*shape))


@pytest.fixture(scope="module")
def mean(slcs):
    amp_stack = np.abs(slcs)
    return np.mean(amp_stack, axis=0)


@pytest.fixture(scope="module")
def var(slcs):
    amp_stack = np.abs(slcs)
    return np.var(amp_stack, axis=0)


def test_shp_glrt(mean, var):
    method = "glrt"

    halfwin_rowcol = (5, 5)  # Looking at the entire stack

    # Use a small alpha
    neighbors = shp.estimate_neighbors(
        mean=mean,
        var=var,
        halfwin_rowcol=halfwin_rowcol,
        nslc=NUM_SLCS,
        alpha=0.005,
        method=method,
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that all 120/121 are neighbors
    assert shps_mid_pixel.sum() == 120

    # TODO: Skip for now, decide if we care about boundaries enough
    # to add back in now that JAX makes it harder for edge checking
    # Use pytest to note the skip:
    pytest.skip()

    # Check the edges are cut off
    top_left = neighbors[0, 0]
    assert top_left[:5, :5].sum() == 5 * 5 - 1
    assert top_left[6:, :].sum() == top_left[:, 6:].sum() == 0


def test_shp_ks(slcs):
    amp_stack = np.abs(slcs)
    halfwin_rowcol = (5, 5)  # Looking at the entire stack

    # First try a tiny alpha
    neighbors = shp.estimate_neighbors(
        amp_stack=amp_stack,
        halfwin_rowcol=halfwin_rowcol,
        nslc=NUM_SLCS,
        alpha=1e-6,
        method="ks",
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    assert shps_mid_pixel.sum() == shps_mid_pixel.size - 1

    # Check the edges are cut off and all zeros
    # NOTE: this is different than TF test since KS is slower
    # TODO if i want them to be exactly the same...
    top_left = neighbors[0, 0]
    assert not top_left[:5, :5].any()

    # Next try a large alpha (should be no neighbors)
    neighbors = shp.estimate_neighbors(
        amp_stack=amp_stack,
        halfwin_rowcol=halfwin_rowcol,
        nslc=NUM_SLCS,
        alpha=1.0,
        method=ShpMethod.KS,
    )
    shps_mid_pixel = neighbors[5, 5]
    assert shps_mid_pixel.sum() == 0  # only itself


def test_shp_half_mean_different(mean, var):
    """Run a test where half the image has different mean"""
    method = ShpMethod.GLRT

    halfwin_rowcol = (5, 5)
    # make the top half different amplitude
    mean2 = mean.copy()
    mean2[:5, :] += 2000
    # For this test, make all variances equal (just checking mean)
    var[:] = var[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors(
        mean=mean2,
        var=var,
        halfwin_rowcol=halfwin_rowcol,
        nslc=NUM_SLCS,
        alpha=0.01,
        method=method,
    )
    shps_mid_pixel = neighbors[5, 5]
    # Check that everything is counted as a neighbor
    top_block = shps_mid_pixel[5:, :]
    assert top_block.sum() == top_block.size - 1
    assert not shps_mid_pixel[:5, :].any()


def test_shp_half_var_different(mean, var):
    """Run a test where half the image has different variance"""
    method = "glrt"

    halfwin_rowcol = (5, 5)
    # make the top half different amplitude
    var2 = var.copy()
    var2[:5, :] += 5000
    # For this test, make all means equal (just checking var)
    mean[:] = mean[5, 5]

    # Even at a small alpha, it should identify the top half as different
    neighbors = shp.estimate_neighbors(
        mean=mean,
        var=var2,
        halfwin_rowcol=halfwin_rowcol,
        nslc=NUM_SLCS,
        alpha=0.01,
        method=method,
    )
    shps_mid_pixel = neighbors[5, 5]
    top_block = shps_mid_pixel[5:, :]
    assert top_block.sum() == top_block.size - 1
    assert not shps_mid_pixel[:5, :].any()


@pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
def test_shp_glrt_nodata_0(mean, var, strides):
    """Ensure"""
    method = "glrt"

    halfwin_rowcol = (5, 5)  # Looking at the entire stack

    mean = mean.copy()
    mean[:4, :4] = 0
    # First try a small alpha
    neighbors = shp.estimate_neighbors(
        mean=mean,
        var=var,
        halfwin_rowcol=halfwin_rowcol,
        strides=strides,
        nslc=NUM_SLCS,
        alpha=0.005,
        method=method,
    )
    out_row, out_col = 2 // strides[0], 2 // strides[1]
    assert neighbors[:out_row, :out_col, :, :].sum() == 0


@pytest.mark.parametrize("method", ["glrt", "ks"])
@pytest.mark.parametrize("alpha", [0.001, 0.005])
@pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
def test_shp_statistics(method, alpha, strides):
    """Check that with repeated tries, the alpha is correct."""

    nsim = 200
    shape = (30, 11, 11)
    halfwin_rowcol = (5, 5)
    shp_counts = np.zeros(nsim)

    amp_stack_sims = rayleigh.rvs(scale=10, size=(nsim, *shape))
    for i in range(nsim):
        amp_stack = amp_stack_sims[i]
        mean = np.mean(amp_stack, axis=0)
        var = np.var(amp_stack, axis=0)
        amp_stack.shape[0]

        neighbors = shp.estimate_neighbors(
            mean=mean,
            var=var,
            halfwin_rowcol=halfwin_rowcol,
            nslc=NUM_SLCS,
            strides=strides,
            alpha=alpha,
            method=method,
            amp_stack=amp_stack,
        )

        out_rows, out_cols = neighbors.shape[:2]
        shps_mid_pixel = neighbors[out_rows // 2, out_cols // 2]
        # dont count center (self) as one
        shp_counts[i] = shps_mid_pixel.sum() - 1

    # Check that the mean number of SHPs is close to 5%
    nbox = shape[1] * shape[2] - 1
    shp_frac = shp_counts.mean() / nbox
    tol_pct = 3
    assert 100 * np.abs(shp_frac - (1 - alpha)) < tol_pct


def test_remove_unconnected():
    # Test 1: All True values
    data = np.ones((5, 5), dtype=bool)
    result = remove_unconnected(data, inplace=False)
    assert np.all(result == data)

    # Resused in all other tests:
    center_only = np.zeros((5, 5), dtype=bool)
    center_only[2, 2] = True

    # Test 2: All False values except center
    data = center_only.copy()
    result = remove_unconnected(data, inplace=False)
    assert np.all(result == center_only)

    # Test 3: Single unconnected True value
    data = np.zeros((5, 5), dtype=bool)
    data[2, 2] = data[0, 0] = True

    result = remove_unconnected(data, inplace=False)
    assert np.all(result == center_only)

    # Test 4: Single unconnected True value with a connected True group
    data = center_only.copy()
    data[3, 3] = data[0, 0] = True

    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = expected[3, 3] = True
    result = remove_unconnected(data, inplace=False)
    assert np.all(result == expected)

    # Test 5: Inplace modification
    data = np.zeros((5, 5), dtype=bool)
    data[2, 2] = data[3, 3] = data[0, 0] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = expected[3, 3] = True
    remove_unconnected(data, inplace=True)
    assert np.all(data == expected)
