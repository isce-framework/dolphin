from math import ceil, floor

import numpy as np
import pytest

from dolphin.phase_link import simulate
from dolphin.phase_link.mle import (
    coh_mat,
    estimate_temp_coh,
    evd,
    full_cov_multilooked,
    mle,
    mle_stack,
)
from dolphin.utils import take_looks

try:
    import cupy as cp

    from dolphin.phase_link import mle_gpu  # noqa

    GPU_AVAILABLE = True
except ImportError:
    print("GPU version not available")
    GPU_AVAILABLE = False

NUM_ACQ = 30
np.random.seed(1234)

# Make sure the GPU versions are correct by making simpler versions:


def form_cov(slc1, slc2, looks):
    num = take_looks(slc1 * slc2.conj(), *looks)
    a1 = take_looks(slc1 * slc1.conj(), *looks)
    a2 = take_looks(slc2 * slc2.conj(), *looks)
    return num / np.sqrt(a1 * a2)


def expected_full_cov(slcs, looks):
    # Manually (slowly) form the covariance matrix to compare to functions
    N = slcs.shape[0]
    rows, cols = take_looks(slcs[0], *looks).shape
    out = np.zeros((rows, cols, N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(i + 1, N):
            out[:, :, i, j] = form_cov(slcs[i], slcs[j], looks)
            out[:, :, j, i] = out[:, :, i, j].conj()
        out[:, :, i, i] = 1.0
    return out


def test_full_cov_cpu(shape=(10, 100, 100), looks=(5, 5)):
    # def test_full_cov_cpu(shape=(10, 5, 5), looks=(5, 5)):
    num_slc, rows, cols = shape
    slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    C1 = expected_full_cov(slcs, looks)
    expected_looked_size = tuple(floor(s / l) for s, l in zip((rows, cols), looks))
    assert C1.shape == (expected_looked_size + (num_slc, num_slc))

    C1_cpu = full_cov_multilooked(slcs, looks)
    np.testing.assert_array_almost_equal(C1, C1_cpu)

    # Check the single pixel function
    rlooks, clooks = looks
    for r in range(expected_looked_size[0]):
        for c in range(expected_looked_size[1]):
            rslice = slice(r * rlooks, (r + 1) * rlooks)
            cslice = slice(c * clooks, (c + 1) * clooks)
            cur_samples = slcs[:, rslice, cslice].reshape(num_slc, -1)
            cur_C = coh_mat(cur_samples)
            np.testing.assert_array_almost_equal(C1[r, c, :, :], cur_C)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_full_cov_gpu(shape=(10, 100, 100), looks=(5, 5)):
    num_slc, rows, cols = shape
    slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    # Get the CPU version for comparison
    C1 = expected_full_cov(slcs, looks)

    d_slcs = cp.asarray(slcs)
    C2_gpu = full_cov_multilooked(d_slcs, looks)
    np.testing.assert_array_almost_equal(C1, C2_gpu.get())

    # Set up the full res version using numba
    d_C3 = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)
    threads_per_block = (16, 16)
    blocks_x = ceil(shape[1] / threads_per_block[0])
    blocks_y = ceil(shape[2] / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    half_window = (looks[1] // 2, looks[0] // 2)
    mle_gpu.estimate_c_gpu[blocks, threads_per_block](d_slcs, half_window, d_C3)
    C3 = d_C3.get()
    assert C3.shape == (rows, cols, num_slc, num_slc)
    C3_sub = C3[2 : -2 : looks[0], 2 : -2 : looks[0]]
    assert C3_sub.shape == C1.shape
    np.testing.assert_array_almost_equal(C1, C3_sub)


def test_full_cov_nans(shape=(10, 100, 100), looks=(5, 5)):
    slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    # Nans for one pixel in all SLCs
    num_slc, _, _ = shape
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, 1, 1] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    coh_mat(slc_samples_nan)

    # Nans for an entire SLC
    slc_stack_nan = slcs.copy()
    slc_stack_nan[1, :, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    with pytest.raises(ZeroDivisionError):
        coh_mat(slc_samples_nan)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_full_cov_nans_gpu(shape=(10, 100, 100), looks=(5, 5)):
    slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    # Get the CPU version for comparison
    C1 = expected_full_cov(slcs, looks)

    d_slcs = cp.asarray(slcs)
    C2_gpu = full_cov_multilooked(d_slcs, looks)
    np.testing.assert_array_almost_equal(C1, C2_gpu.get())

    # Set up the full res version using numba
    num_slc, rows, cols = shape
    d_C3 = cp.zeros((rows, cols, num_slc, num_slc), dtype=np.complex64)
    threads_per_block = (16, 16)
    blocks_x = ceil(shape[1] / threads_per_block[0])
    blocks_y = ceil(shape[2] / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    half_window = (looks[1] // 2, looks[0] // 2)
    mle_gpu.estimate_c_gpu[blocks, threads_per_block](d_slcs, half_window, d_C3)
    C3 = d_C3.get()
    assert C3.shape == (rows, cols, num_slc, num_slc)
    C3_sub = C3[2 : -2 : looks[0], 2 : -2 : looks[0]]
    assert C3_sub.shape == C1.shape


# CPU versions of the MLE and EVD estimates
@pytest.fixture
def C_hat(slc_samples):
    return coh_mat(slc_samples)


@pytest.fixture
def est_mle_cpu(C_hat):
    return np.angle(mle(C_hat))


@pytest.fixture
def est_evd_cpu(C_hat):
    return np.angle(evd(C_hat))


# Check that the estimates are close to the truth
def test_estimation(C_truth, est_mle_cpu, est_evd_cpu):
    _, truth = C_truth

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_evd_cpu)) < err_deg
    assert np.degrees(simulate.rmse(truth, est_mle_cpu)) < err_deg


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_estimation_gpu(slc_samples, est_mle_cpu):
    # Get the GPU version
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    est_mle_gpu_ml, temp_coh = mle_gpu.run_mle_multilooked_gpu(
        slc_stack, half_window=(5, 5)
    )
    assert est_mle_gpu_ml.shape == (len(est_mle_cpu), 1, 1)
    assert temp_coh.shape == (1, 1)
    est_phase_gpu = np.angle(np.squeeze(est_mle_gpu_ml))
    np.testing.assert_array_almost_equal(est_mle_cpu, est_phase_gpu, decimal=3)

    est_mle_gpu_fullres, temp_coh = mle_gpu.run_mle_gpu(slc_stack, half_window=(5, 5))
    assert est_mle_gpu_fullres.shape == (len(est_mle_cpu), 11, 11)
    assert temp_coh.shape == (11, 11)
    # The middle pixel should be the same, since it had the full window
    est_phase_gpu2 = np.angle(est_mle_gpu_fullres[:, 5, 5])
    np.testing.assert_array_almost_equal(est_mle_cpu, est_phase_gpu2, decimal=3)


def test_mask(slc_samples, C_truth):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    # slc_samples_masked = slc_stack[:, 1:, :].reshape(NUM_ACQ, -1)
    slc_samples_masked = slc_stack[:, ~mask]

    _, truth = C_truth
    C_hat = coh_mat(slc_samples_masked)
    est_mle = np.angle(mle(C_hat))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan

    # take_looks should ignore nans
    C_full = full_cov_multilooked(slc_stack_masked, looks=(11, 11))
    np.testing.assert_array_almost_equal(np.squeeze(C_full), C_hat)

    est_multilooked = np.squeeze(mle_stack(C_full))
    np.testing.assert_array_almost_equal(est_mle, est_multilooked, decimal=1)

    # if not GPU_AVAILABLE:
    #     pytest.skip("GPU version not available")
    # # Now check both GPU versions


def test_temp_coh():
    sigmas = [0.01, 0.1, 1, 10]
    expected_tcoh_bounds = [(0.99, 1), (0.9, 1.0), (0.3, 0.5), (0.0, 0.3)]
    out_tc = []
    out_C = []
    out_truth = []
    for sigma, (t_low, t_high) in zip(sigmas, expected_tcoh_bounds):
        C, truth = simulate.simulate_C(
            num_acq=30,
            Tau0=72,
            gamma_inf=0.95,
            gamma0=0.99,
            add_signal=True,
            signal_std=sigma,
        )
        # Get the explicit for-loop version
        temp_coh = simulate.estimate_temp_coh(np.exp(1j * truth), C)
        assert t_low <= temp_coh <= t_high
        out_tc.append(temp_coh)
        out_C.append(C)
        out_truth.append(truth)

    # Test the block version of the function
    # Just repeat the first C matrix and first estimate
    C_arrays = np.array(out_C).reshape((2, 2, 30, 30))
    est_arrays = np.exp(1j * np.array(out_truth).reshape((2, 2, 30)))
    # mimic the shape of (nslc, rows, cols)
    est_arrays = np.moveaxis(est_arrays, -1, 0)
    tc_image = estimate_temp_coh(est_arrays, C_arrays)

    expected_tc_image = np.array(out_tc).reshape((2, 2))
    np.testing.assert_array_almost_equal(tc_image, expected_tc_image)
