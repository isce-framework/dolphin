import os

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _core, covariance, simulate
from dolphin.phase_link._ps_filling import fill_ps_pixels
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"
NUM_ACQ = 30
simulate._seed(1234)

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


@pytest.fixture(scope="module")
def slc_samples(C_truth):
    C, _ = C_truth
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


# CPU versions of the MLE and EVD estimates
@pytest.fixture(scope="module")
def C_hat(slc_samples):
    return np.array(covariance.coh_mat_single(slc_samples))


# Make the single-pixel comparisons with simple implementation
@pytest.fixture(scope="module")
def est_mle_verify(C_hat):
    return np.angle(simulate.mle(C_hat))


@pytest.fixture(scope="module")
def est_evd_verify(C_hat):
    return np.angle(simulate.evd(C_hat))


@pytest.mark.parametrize("use_evd", [False, True])
def test_estimation(C_truth, slc_samples, est_mle_verify, est_evd_verify, use_evd):
    _, truth = C_truth

    # Check that the estimates are close to the truth
    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_evd_verify)) < err_deg
    assert np.degrees(simulate.rmse(truth, est_mle_verify)) < err_deg

    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)

    est_mle_fullres, temp_coh, eigs, _ = _core.run_cpl(
        slc_stack, HalfWindow(x=5, y=5), Strides(x=1, y=1), use_evd=use_evd
    )
    assert est_mle_fullres.shape == (len(est_mle_verify), 11, 11)
    assert temp_coh.shape == (11, 11)
    assert eigs.shape == (11, 11)
    assert np.all(eigs > 0)
    # The middle pixel should be the same, since it had the full window
    est_phase = est_mle_fullres[:, 5, 5]
    npt.assert_array_almost_equal(est_mle_verify, est_phase, decimal=3)


def test_masked(slc_samples, C_truth):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    # slc_samples_masked = slc_stack[:, 1:, :].reshape(NUM_ACQ, -1)
    slc_samples_masked = slc_stack[:, ~mask]

    _, truth = C_truth
    C_hat1 = covariance.coh_mat_single(slc_samples_masked)
    est_mle = np.angle(simulate.mle(np.array(C_hat1)))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan

    C_hat2 = covariance.estimate_stack_covariance(
        slc_stack_masked, half_window=HalfWindow(5, 5)
    )
    est_full = _core.process_coherence_matrices(C_hat2)[0]
    # Middle pixel should be the same
    npt.assert_array_almost_equal(est_mle, est_full[:, 5, 5], decimal=1)


def test_run_phase_linking(slc_samples):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mle_est, _, _, _ = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
    )

    C_hat = covariance.coh_mat_single(slc_samples)
    expected_phase = np.angle(simulate.mle(np.array(C_hat)))

    # Middle pixel should be the same
    npt.assert_array_almost_equal(expected_phase, np.angle(mle_est[:, 5, 5]), decimal=1)


def test_run_phase_linking_norm_output(slc_samples):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[1, 1] = True
    mle_est, _, _, _ = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        ps_mask=ps_mask,
        use_slc_amp=False,
    )
    assert np.allclose(np.abs(mle_est), 1)


@pytest.mark.parametrize("strides", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("half_window", [5, 11])
@pytest.mark.parametrize("shape", [(10, 11), (15, 20), (50, 75)])
def test_strides_window_sizes(strides, half_window, shape):
    """Test that window and stride sizes don't cause errors for many input shapes."""
    data = np.random.normal(0, 1, size=(4, *shape)).astype(np.complex64)
    if 2 * half_window + 1 > shape[0]:
        # TODO: We should raise an appropriate error, then catch here
        pytest.skip("Window size is too large for input shape")
    _core.run_phase_linking(
        data,
        half_window=HalfWindow(half_window, half_window),
        strides=Strides(strides, strides),
    )


@pytest.mark.parametrize("strides", [1, 2, 3, 4])
def test_ps_fill(slc_samples, strides):
    rows, cols = 11, 11
    slc_stack = slc_samples.reshape(NUM_ACQ, rows, cols)

    mle_est = np.zeros((NUM_ACQ, rows // strides, cols // strides), dtype=np.complex64)
    temp_coh = np.zeros(mle_est.shape[1:])

    ps_idx = 2
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True

    fill_ps_pixels(
        mle_est,
        temp_coh,
        slc_stack,
        ps_mask,
        Strides(strides, strides),
        None,  # avg_mag
    )

    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(mle_est[:, out_idx, out_idx])
    )

    assert temp_coh[out_idx, out_idx] == 1


@pytest.mark.parametrize("strides", [1, 2, 3])
def test_run_phase_linking_ps_fill(slc_samples, strides):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    ps_idx = 2
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True
    mle_est, temp_coh, _, _ = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        strides=Strides(strides, strides),
        ps_mask=ps_mask,
    )
    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(mle_est[:, out_idx, out_idx])
    )

    assert temp_coh[out_idx, out_idx] == 1
