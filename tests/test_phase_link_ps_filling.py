from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import Strides
from dolphin.phase_link import simulate
from dolphin.phase_link._ps_filling import fill_ps_pixels, get_max_idxs

RNG = np.random.default_rng()


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


def test_get_max_idxs_uneven_shape():
    arr = np.arange(9 * 11).reshape(9, 11)
    rows, cols = get_max_idxs(arr, 2, 3)
    expected = np.array([[13, 16, 19], [35, 38, 41], [57, 60, 63], [79, 82, 85]])
    npt.assert_array_equal(arr[rows, cols], expected.ravel())


def test_failing_ps_size():
    data = np.load(Path(__file__).parent / "data/ps-fix/failing_data_idxs.npz")
    shape2d = (522, 534)
    slc_stack = RNG.normal(size=(15, *shape2d)) + 1j * RNG.normal(size=(15, *shape2d))
    slc_stack = slc_stack.astype("complex64")
    slc_stack[data["slc_nan_t"], data["slc_nan_y"], data["slc_nan_x"]] = np.nan
    reference_idx = 0

    mags = RNG.normal(size=shape2d) ** 2
    avg_mag = np.full_like(mags, np.nan)
    rows, cols = data["not_nan_rows"], data["not_nan_cols"]
    avg_mag[rows, cols] = mags[rows, cols]
    ps_mask = np.zeros(shape2d, dtype=bool)
    ps_mask[rows, cols] = True

    small_shape2d = (174, 89)
    temp_coh = RNG.random(size=small_shape2d)
    strides = Strides(3, 6)
    cpx_phase = RNG.normal(size=(15, *small_shape2d)) + 1j * RNG.normal(
        size=(15, *small_shape2d)
    )

    fill_ps_pixels(
        cpx_phase=cpx_phase,
        temp_coh=temp_coh,
        slc_stack=slc_stack,
        strides=strides,
        avg_mag=avg_mag,
        ps_mask=ps_mask,
        reference_idx=reference_idx,
    )
