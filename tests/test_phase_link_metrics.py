import numpy as np
import numpy.testing as npt

from dolphin.phase_link import metrics, simulate


def _estimate_temp_coh(est, cov_matrix):
    """explicit version of one pixel's temporal coherence"""
    gamma = 0
    N = len(est)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            theta = np.angle(cov_matrix[i, j])
            phi = np.angle(est[i] * np.conj(est[j]))

            gamma += np.exp(1j * theta) * np.exp(-1j * phi)
            count += 1
    return np.abs(gamma) / count


def test_temp_coh():
    sigmas = [0.01, 0.1, 1, 10]
    expected_tcoh_bounds = [(0.99, 1), (0.9, 1.0), (0.6, 0.9), (0.0, 0.3)]
    out_tc = []
    out_C = []
    out_truth = []
    for sigma, (t_low, t_high) in zip(sigmas, expected_tcoh_bounds):
        simulate._seed(1)
        C, truth = simulate.simulate_coh(
            num_acq=30,
            Tau0=72,
            gamma_inf=0.95,
            gamma0=0.99,
            add_signal=True,
            signal_std=sigma,
        )
        # Get the explicit for-loop version
        temp_coh = _estimate_temp_coh(np.exp(1j * truth), C)
        assert t_low <= temp_coh <= t_high
        out_tc.append(temp_coh)
        out_C.append(C)
        out_truth.append(truth)

    # Test passing one pixel's worth of data
    tc_pixel = metrics.estimate_temp_coh(np.exp(1j * truth), C)
    assert np.allclose(tc_pixel, temp_coh)

    # Test the block version of the function
    # Just repeat the first C matrix and first estimate
    C_arrays = np.array(out_C).reshape((2, 2, 30, 30))
    est_arrays = np.exp(1j * np.array(out_truth).reshape((2, 2, 30)))
    tc_image = metrics.estimate_temp_coh(est_arrays, C_arrays)

    expected_tc_image = np.array(out_tc).reshape((2, 2))
    npt.assert_array_almost_equal(tc_image, expected_tc_image)
