import numpy as np
import numpy.testing as npt
import pytest

from dolphin.phase_link import covariance, mle, simulate
from dolphin.phase_link._mle_gpu import run_gpu
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available()
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
    return covariance.coh_mat_single(slc_samples)


# Make the single-pixel comparisons with simple implementation
@pytest.fixture(scope="module")
def est_mle_cpu(C_hat):
    return np.angle(simulate.mle(C_hat))


@pytest.fixture(scope="module")
def est_evd_cpu(C_hat):
    return np.angle(simulate.evd(C_hat))


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

    est_mle_gpu_fullres, temp_coh = run_gpu(slc_stack, half_window={"x": 5, "y": 5})
    assert est_mle_gpu_fullres.shape == (len(est_mle_cpu), 11, 11)
    assert temp_coh.shape == (11, 11)
    # The middle pixel should be the same, since it had the full window
    est_phase_gpu2 = np.angle(est_mle_gpu_fullres[:, 5, 5])
    npt.assert_array_almost_equal(est_mle_cpu, est_phase_gpu2, decimal=3)


def test_masked(slc_samples, C_truth):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    # slc_samples_masked = slc_stack[:, 1:, :].reshape(NUM_ACQ, -1)
    slc_samples_masked = slc_stack[:, ~mask]

    _, truth = C_truth
    C_hat1 = covariance.coh_mat_single(slc_samples_masked)
    est_mle = np.angle(simulate.mle(C_hat1))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan

    C_hat2 = covariance.estimate_stack_covariance_cpu(
        slc_stack_masked, half_window={"x": 5, "y": 5}
    )
    est_full = np.squeeze(mle.mle_stack(C_hat2))
    # Middle pixel should be the same
    npt.assert_array_almost_equal(est_mle, est_full[:, 5, 5], decimal=1)

    if not GPU_AVAILABLE:
        pytest.skip("GPU version not available")
    # Now check GPU version
    est_mle_gpu_fullres, temp_coh = run_gpu(
        slc_stack_masked, half_window={"x": 5, "y": 5}
    )
    est_phase_gpu = np.angle(est_mle_gpu_fullres[:, 5, 5])
    npt.assert_array_almost_equal(est_mle, est_phase_gpu, decimal=1)


def test_run_mle(slc_samples):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    mle_est, _ = mle.run_mle(
        slc_stack,
        half_window={"x": 5, "y": 5},
        gpu_enabled=False,
    )

    C_hat = covariance.coh_mat_single(slc_samples)
    expected_phase = np.angle(simulate.mle(C_hat))

    # Middle pixel should be the same
    npt.assert_array_almost_equal(expected_phase, np.angle(mle_est[:, 5, 5]), decimal=1)


def test_run_mle_norm_output(slc_samples):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[1, 1] = True
    mle_est, _ = mle.run_mle(
        slc_stack,
        half_window={"x": 5, "y": 5},
        ps_mask=ps_mask,
        use_slc_amp=False,
        gpu_enabled=False,
    )
    assert np.allclose(np.abs(mle_est), 1)


@pytest.mark.parametrize("gpu_enabled", [True, False])
@pytest.mark.parametrize("strides", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("half_window", [5, 11])
@pytest.mark.parametrize("shape", [(10, 11), (15, 20), (50, 75)])
def test_strides_window_sizes(gpu_enabled, strides, half_window, shape):
    """Test that window and stride sizes don't cause errors for many input shapes."""
    data = np.random.normal(0, 1, size=(4, *shape)).astype(np.complex64)
    mle.run_mle(
        data,
        half_window={"x": half_window, "y": half_window},
        strides={"x": strides, "y": strides},
        gpu_enabled=gpu_enabled,
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

    mle._fill_ps_pixels(
        mle_est,
        temp_coh,
        slc_stack,
        ps_mask,
        {"x": strides, "y": strides},
        None,  # avg_mag
    )

    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(mle_est[:, out_idx, out_idx])
    )

    assert temp_coh[out_idx, out_idx] == 1


@pytest.mark.parametrize("gpu_enabled", [True, False])
@pytest.mark.parametrize("strides", [1, 2, 3])
def test_run_mle_ps_fill(slc_samples, gpu_enabled, strides):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    ps_idx = 2
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True
    mle_est, temp_coh = mle.run_mle(
        slc_stack,
        half_window={"x": 5, "y": 5},
        strides={"x": strides, "y": strides},
        ps_mask=ps_mask,
        gpu_enabled=gpu_enabled,
    )
    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(mle_est[:, out_idx, out_idx])
    )

    assert temp_coh[out_idx, out_idx] == 1
