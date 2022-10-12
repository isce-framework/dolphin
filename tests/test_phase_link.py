from math import ceil

import numpy as np
import pytest

from dolphin import phase_link
from dolphin.phase_link import simulate
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


def full_cov(slcs, looks):
    N = slcs.shape[0]
    rows, cols = take_looks(slcs[0], *looks).shape
    out = np.zeros((rows, cols, N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(i + 1, N):
            out[:, :, i, j] = form_cov(slcs[i], slcs[j], looks)
            out[:, :, j, i] = out[:, :, i, j].conj()
        out[:, :, i, i] = 1.0
    return out


def test_full_cov(shape=(10, 100, 100), looks=(5, 5)):
    # import time

    num_slc, rows, cols = shape
    slcs = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    # t0 = time.time()
    C1 = full_cov(slcs, looks)
    # t1 = time.time()
    C2_cpu = phase_link.full_cov_multilooked(slcs, looks)
    # t2 = time.time()
    # print(f"CPU: {t1 - t0:.2f}, GPU: {t2 - t1:.2f} seconds")
    np.testing.assert_array_almost_equal(C1, C2_cpu)

    if not GPU_AVAILABLE:
        pytest.skip("GPU version not available")

    d_slcs = cp.asarray(slcs)
    C2_gpu = phase_link.full_cov_multilooked(d_slcs, looks)
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


@pytest.fixture
def C_hat(slc_samples):
    return phase_link.coh_mat(slc_samples)


# @pytest.mark.skip
def test_estimation(C_truth, C_hat):
    _, truth = C_truth

    est_evd = np.angle(phase_link.evd(C_hat))
    est_mle = np.angle(phase_link.mle(C_hat))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_evd)) < err_deg
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg


# @pytest.mark.skip
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_estimation_gpu(slc_samples, C_hat):
    # Calc the CPU version
    est_mle = np.angle(phase_link.mle(C_hat))

    # Get the GPU version
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    est_mle_gpu_ml = mle_gpu.run_mle_multilooked_gpu(slc_stack, half_window=(5, 5))
    assert est_mle_gpu_ml.shape == (len(est_mle), 1, 1)
    est_phase_gpu = np.angle(np.squeeze(est_mle_gpu_ml))
    np.testing.assert_array_almost_equal(est_mle, est_phase_gpu, decimal=3)

    est_mle_gpu_fullres = mle_gpu.run_mle_gpu(slc_stack, half_window=(5, 5))
    assert est_mle_gpu_fullres.shape == (len(est_mle), 11, 11)
    # The middle pixel should be the same, since it had the full window
    est_phase_gpu2 = np.angle(est_mle_gpu_fullres[:, 5, 5])
    np.testing.assert_array_almost_equal(est_mle, est_phase_gpu2, decimal=3)


def test_mask(slc_samples, C_truth):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    # slc_samples_masked = slc_stack[:, 1:, :].reshape(NUM_ACQ, -1)
    slc_samples_masked = slc_stack[:, ~mask]

    _, truth = C_truth
    C_hat = phase_link.coh_mat(slc_samples_masked)
    est_mle = np.angle(phase_link.mle(C_hat))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan

    # take_looks should ignore nans
    C_full = phase_link.full_cov_multilooked(slc_stack_masked, looks=(11, 11))
    np.testing.assert_array_almost_equal(np.squeeze(C_full), C_hat)

    est_multilooked = np.squeeze(phase_link.mle_stack(C_full))
    np.testing.assert_array_almost_equal(est_mle, est_multilooked, decimal=1)

    # if not GPU_AVAILABLE:
    #     pytest.skip("GPU version not available")
    # # Now check both GPU versions
