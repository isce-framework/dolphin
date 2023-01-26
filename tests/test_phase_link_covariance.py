from math import ceil, floor

import numpy as np
import numpy.testing as npt
import pytest

from dolphin.io import compute_out_shape
from dolphin.phase_link import covariance, simulate
from dolphin.utils import gpu_is_available, take_looks

GPU_AVAILABLE = gpu_is_available()
NUM_ACQ = 30
simulate._seed(1234)

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


# Make sure the GPU versions are correct by making simpler versions:
def form_cov(slc1, slc2, looks):
    num = take_looks(slc1 * slc2.conj(), *looks)
    a1 = take_looks(slc1 * slc1.conj(), *looks)
    a2 = take_looks(slc2 * slc2.conj(), *looks)
    return num / np.sqrt(a1 * a2)


def get_expected_cov(slcs, looks):
    # Manually (slowly) form the covariance matrix at all pixels
    N = slcs.shape[0]
    rows, cols = take_looks(slcs[0], *looks).shape
    out = np.zeros((rows, cols, N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(i + 1, N):
            out[:, :, i, j] = form_cov(slcs[i], slcs[j], looks)
            out[:, :, j, i] = out[:, :, i, j].conj()
        out[:, :, i, i] = 1.0
    return out


@pytest.fixture(scope="module")
def slcs(shape=(10, 100, 100)):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


@pytest.fixture
def expected_cov(slcs, looks=(5, 5)):
    return get_expected_cov(slcs, looks)


def test_coh_mat_single(slcs, expected_cov, looks=(5, 5)):
    num_slc, rows, cols = slcs.shape

    # Check the single pixel function
    expected_looked_size = tuple(floor(s / l) for s, l in zip((rows, cols), looks))
    assert expected_cov.shape == (expected_looked_size + (num_slc, num_slc))

    r_looks, c_looks = looks
    for r in range(expected_looked_size[0]):
        for c in range(expected_looked_size[1]):
            r_slice = slice(r * r_looks, (r + 1) * r_looks)
            c_slice = slice(c * c_looks, (c + 1) * c_looks)
            cur_samples = slcs[:, r_slice, c_slice].reshape(num_slc, -1)
            cur_C = covariance.coh_mat_single(cur_samples)
            npt.assert_array_almost_equal(expected_cov[r, c, :, :], cur_C)


def test_estimate_stack_covariance_cpu(slcs, expected_cov, looks=(5, 5)):
    # Check the full stack function
    r_looks, c_looks = looks
    half_window = {"x": c_looks // 2, "y": r_looks // 2}
    strides = {"x": c_looks, "y": r_looks}
    C1_cpu = covariance.estimate_stack_covariance_cpu(
        slcs, half_window=half_window, strides=strides
    )
    npt.assert_array_almost_equal(expected_cov, C1_cpu)

    # Check multi-processing
    C1_cpu_mp = covariance.estimate_stack_covariance_cpu(
        slcs, half_window=half_window, strides=strides, n_workers=2
    )
    npt.assert_array_almost_equal(expected_cov, C1_cpu_mp)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_estimate_stack_covariance_gpu(slcs, expected_cov, looks=(5, 5)):
    import cupy as cp

    num_slc, rows, cols = slcs.shape
    # Get the CPU version for comparison
    expected_cov = get_expected_cov(slcs, looks)

    # Set up the full res version using numba
    d_slcs = cp.asarray(slcs)

    strides = {"x": 1, "y": 1}
    out_rows, out_cols = rows, cols
    d_C3 = cp.zeros((out_rows, out_cols, num_slc, num_slc), dtype=np.complex64)
    threads_per_block = (16, 16)
    blocks_x = ceil(slcs.shape[1] / threads_per_block[0])
    blocks_y = ceil(slcs.shape[2] / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    half_rowcol = (looks[0] // 2, looks[1] // 2)
    strides_rowcol = (strides["y"], strides["x"])

    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slcs, half_rowcol, strides_rowcol, d_C3
    )
    C3 = d_C3.get()
    # assert C3.shape == (out_rows, out_cols, num_slc, num_slc)
    C3_sub = C3[2 : -2 : looks[0], 2 : -2 : looks[0]]
    assert C3_sub.shape == expected_cov.shape
    npt.assert_array_almost_equal(expected_cov, C3_sub)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_estimate_stack_covariance_gpu_strides(slcs, expected_cov, looks=(5, 5)):
    import cupy as cp

    num_slc, rows, cols = slcs.shape
    # Get the CPU version for comparison
    expected_cov = get_expected_cov(slcs, looks)

    # Set up the full res version using numba
    d_slcs = cp.asarray(slcs)

    r_looks, c_looks = looks
    strides = {"x": c_looks, "y": r_looks}
    out_rows, out_cols = compute_out_shape((rows, cols), strides)
    d_C3 = cp.zeros((out_rows, out_cols, num_slc, num_slc), dtype=np.complex64)
    threads_per_block = (16, 16)
    blocks_x = ceil(slcs.shape[1] / threads_per_block[0])
    blocks_y = ceil(slcs.shape[2] / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    half_rowcol = (looks[0] // 2, looks[1] // 2)
    strides_rowcol = (strides["y"], strides["x"])
    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slcs, half_rowcol, strides_rowcol, d_C3
    )
    # Now this should be the same size as the multi-looked version
    C3 = d_C3.get()
    assert C3.shape == expected_cov.shape
    npt.assert_array_almost_equal(expected_cov, C3)


def test_estimate_stack_covariance_nans(slcs):
    num_slc, _, _ = slcs.shape

    C = covariance.coh_mat_single(slcs.reshape(num_slc, -1))
    # Nans for one pixel in all SLCs
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, 1, 1] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    C_nan = covariance.coh_mat_single(slc_samples_nan)
    assert np.max(np.abs(C - C_nan)) < 0.01

    # Nans for an entire SLC
    slc_stack_nan = slcs.copy()
    slc_stack_nan[1, :, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    # TODO: should we raise an error if we pass a dead SLC?
    # with pytest.raises(ZeroDivisionError):
    assert covariance.coh_mat_single(slc_samples_nan)[0, 1] == 0
    assert covariance.coh_mat_single(slc_samples_nan)[1, 0] == 0

    # all nans should return an identity matrix (no correlation anywhere)
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, :, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    assert covariance.coh_mat_single(slc_samples_nan).sum() == num_slc


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_estimate_stack_covariance_nans_gpu(slcs, looks=(5, 5)):
    import cupy as cp

    # Set up the full res version using numba
    num_slc, rows, cols = slcs.shape
    threads_per_block = (16, 16)
    blocks_x = ceil(rows / threads_per_block[0])
    blocks_y = ceil(cols / threads_per_block[1])
    blocks = (blocks_x, blocks_y)
    d_slcs = cp.asarray(slcs)
    d_C = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)

    half_rowcol = (looks[0] // 2, looks[1] // 2)
    strides_rowcol = (1, 1)
    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slcs, half_rowcol, strides_rowcol, d_C
    )
    C_nonan = d_C.get()

    slcs_nan = slcs.copy()
    slcs_nan[:, 20, 20] = np.nan
    d_slcs_nan = cp.asarray(slcs_nan)
    d_C_nan = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)
    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slcs_nan, half_rowcol, strides_rowcol, d_C_nan
    )
    C_nan = d_C_nan.get()

    # Make sure it doesn't affect pixels far away
    assert np.abs(C_nonan[5, 5] - C_nan[5, 5]).max() < 1e-6
    # Should still be close to the non-nan version
    assert np.max(np.abs(C_nonan - C_nan)) < 0.10
