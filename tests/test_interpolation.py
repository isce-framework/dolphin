import numpy as np

from dolphin import interpolation


def test_interpolate():
    x, y = np.meshgrid(np.arange(200), np.arange(100))
    # simulate a simple phase ramp
    phase = 0.003 * x + 0.002 * y
    # interferogram with the simulated phase ramp and with a constant amplitude
    ifg = np.exp(1j * phase)
    corr = np.ones(ifg.shape)
    # mask out the ifg/corr at a given pixel for example at pixel 50,40
    # index of the pixel of interest
    x_idx = 50
    y_idx = 40
    corr[y_idx, x_idx] = 0

    interpolated_ifg = np.zeros((100, 200), dtype=np.complex64)
    interpolation.interpolate(
        ifg,
        weights=corr,
        weight_cutoff=0.5,
        num_neighbors=20,
        alpha=0.75,
        max_radius=51,
        min_radius=0,
    )
    # expected phase based on the model above used for simulation
    expected_phase = 0.003 * x_idx + 0.002 * y_idx
    phase_error = np.angle(
        interpolated_ifg[y_idx, x_idx] * np.exp(-1j * expected_phase)
    )
    assert np.allclose(phase_error, 0.0, atol=1e-3)
