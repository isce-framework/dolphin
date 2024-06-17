import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import Strides
from dolphin.phase_link import simulate
from dolphin.phase_link._ps_filling import fill_ps_pixels, get_max_idxs

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


def test_get_max_idxs_no_look():
    arr = np.array([[5, 1, 3], [4, 9, 2], [8, 7, 6]])
    row_looks = 1
    col_looks = 1
    expected_rows, expected_cols = np.where(arr == arr)
    actual_rows, actual_cols = get_max_idxs(arr, row_looks, col_looks)
    assert np.array_equal(actual_rows, expected_rows)
    assert np.array_equal(actual_cols, expected_cols)


def test_get_max_idxs_2x2_window():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    row_looks = 2
    col_looks = 2
    expected_rows = np.array([1, 1, 3, 3])
    expected_cols = np.array([1, 3, 1, 3])
    actual_rows, actual_cols = get_max_idxs(arr, row_looks, col_looks)
    assert np.array_equal(actual_rows, expected_rows)
    assert np.array_equal(actual_cols, expected_cols)


def test_get_max_idxs_3x3_mixed_values():
    arr = np.array([[1, 5, 3], [4, 20, 2], [8, 0, 6]])
    row_looks = 2
    col_looks = 2
    expected_rows = np.array([1])
    expected_cols = np.array([1])
    actual_rows, actual_cols = get_max_idxs(arr, row_looks, col_looks)
    assert np.array_equal(actual_rows, expected_rows)
    assert np.array_equal(actual_cols, expected_cols)


def generate_manual_expected_indices(n, looks):
    # For the simple reshaping of arange to a rectangle, the max values are
    # the bottom right corners of each window.
    step = looks
    num_windows = n // looks
    end_idx = looks - 1
    rows = np.repeat(np.arange(end_idx, step * num_windows, step), num_windows)
    cols = np.tile(np.arange(end_idx, step * num_windows, step), num_windows)
    return rows, cols


@pytest.mark.parametrize("n", list(range(10, 15)))
@pytest.mark.parametrize("looks", [2, 3])
def test_get_max_idxs_param(n, looks):
    arr = np.arange(n * n).reshape(n, n)
    expected = generate_manual_expected_indices(n, looks)
    actual_rows, actual_cols = get_max_idxs(arr, looks, looks)
    assert np.array_equal(actual_rows, expected[0])
    assert np.array_equal(actual_cols, expected[1])
