# import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import Strides
from dolphin.phase_link import simulate
from dolphin.phase_link._ps_filling import fill_ps_pixels

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


@pytest.mark.parametrize("use_max_ps", [True, False])
@pytest.mark.parametrize("strides", [1, 2, 3, 4])
@pytest.mark.parametrize("rows", list(range(10, 16)))
@pytest.mark.parametrize("cols", list(range(10, 12)))
def test_ps_fill(C_truth, use_max_ps, strides, rows, cols):
    """Test substituting the PS pixel phases into the phase linking result."""
    num_acq = 9
    C = C_truth[0][:num_acq, :num_acq]

    slc_stack = simulate.simulate_neighborhood_stack(C, rows * cols).reshape(
        num_acq, rows, cols
    )

    # Fake an all zero-phase solution that will get filled with the PS pixel phases
    pl_est = np.ones((num_acq, rows // strides, cols // strides), dtype=np.complex64)
    temp_coh = np.zeros(pl_est.shape[1:])

    ps_idx = 2
    ps_mask = np.zeros((rows, cols), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True

    fill_ps_pixels(
        pl_est,
        temp_coh,
        slc_stack,
        ps_mask,
        Strides(strides, strides),
        None,  # avg_mag
        use_max_ps=use_max_ps,
    )

    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(pl_est[:, out_idx, out_idx])
    )

    assert temp_coh[out_idx, out_idx] == 1
