import os

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _core, covariance, simulate
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"
NUM_ACQ = 30
simulate._seed(1234)


@pytest.fixture(scope="module")
def slc_samples(C_truth):
    C, _ = C_truth
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


@pytest.mark.parametrize("baseline_lag", [None, 5])
@pytest.mark.parametrize("use_evd", [False, True])
def test_estimation(C_truth, slc_samples, use_evd, baseline_lag):
    _, truth = C_truth

    C_hat = np.array(covariance.coh_mat_single(slc_samples))
    # Make the single-pixel comparisons with simple implementation
    est_mle_verify = np.angle(simulate.mle(C_hat))
    est_evd_verify = np.angle(simulate.evd(C_hat))

    # Check that the estimates are close to the truth
    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_evd_verify)) < err_deg
    assert np.degrees(simulate.rmse(truth, est_mle_verify)) < err_deg

    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)

    # cpx_phase, temp_coh, eigs, _ = _core.run_cpl(
    pl_out = _core.run_cpl(
        slc_stack,
        HalfWindow(x=5, y=5),
        Strides(x=1, y=1),
        use_evd=use_evd,
        baseline_lag=baseline_lag,
    )
    assert pl_out.cpx_phase.shape == (len(est_mle_verify), 11, 11)
    assert pl_out.temp_coh.shape == (11, 11)
    assert pl_out.eigenvalues.shape == (11, 11)
    if use_evd:
        expected_min_eig = baseline_lag if baseline_lag else NUM_ACQ / 3
        assert np.all(pl_out.eigenvalues > expected_min_eig)
        assert pl_out
    else:
        # should be 1, but floating point rounding sometimes drops
        assert np.all(pl_out.eigenvalues > 0.99)

    # The middle pixel should be the same, since it had the full window
    est_phase = np.angle(pl_out.cpx_phase[:, 5, 5])
    npt.assert_array_almost_equal(est_mle_verify, est_phase, decimal=1)


def test_masked(slc_samples, C_truth):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
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
    npt.assert_array_almost_equal(est_mle, np.angle(est_full[5, 5, :]), decimal=1)


def test_run_phase_linking(slc_samples):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
    )

    C_hat = covariance.coh_mat_single(slc_samples)
    expected_phase = np.angle(simulate.mle(np.array(C_hat)))

    # Middle pixel should be the same
    npt.assert_array_almost_equal(
        expected_phase, np.angle(pl_out.cpx_phase[:, 5, 5]), decimal=1
    )


@pytest.mark.parametrize("use_slc_amp", [False, True])
@pytest.mark.parametrize("use_max_ps", [False, True])
def test_run_phase_linking_use_slc_amp(slc_samples, use_slc_amp, use_max_ps):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    ps_mask = np.zeros((11, 11), dtype=bool)
    # Specify at least 1 ps
    ps_mask[1, 1] = True
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        ps_mask=ps_mask,
        use_slc_amp=use_slc_amp,
        use_max_ps=use_max_ps,
    )

    expected = np.abs(slc_stack) if use_slc_amp else 1
    assert np.allclose(np.abs(pl_out.cpx_phase), expected)


def test_run_phase_linking_with_shift(slc_samples):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    # Pretend there's a shift which should lead to nodata at the intersection
    # First row of the first image
    slc_stack[0, 0, :] = np.nan
    # last row of the second image
    slc_stack[1, -1, :] = np.nan
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
    )

    assert np.isnan(pl_out.cpx_phase[:, [0, -1], :]).all()
    assert np.isnan(pl_out.temp_coh[[0, -1], :]).all()

    assert np.all(~np.isnan(pl_out.cpx_phase[:, 1:-1, :]))
    assert np.all(~np.isnan(pl_out.temp_coh[1:-1, :]))


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


@pytest.mark.parametrize("use_max_ps", [True, False])
@pytest.mark.parametrize("strides", [1, 2, 3])
def test_run_phase_linking_ps_fill(slc_samples, use_max_ps, strides):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    ps_idx = 2
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True
    # Ignore RuntimeWarning
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        strides=Strides(strides, strides),
        ps_mask=ps_mask,
        use_max_ps=use_max_ps,
    )
    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(pl_out.cpx_phase[:, out_idx, out_idx])
    )

    assert pl_out.temp_coh[out_idx, out_idx] == 1
