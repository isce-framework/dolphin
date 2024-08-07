import os
from math import floor

import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import covariance, simulate
from dolphin.utils import gpu_is_available, take_looks

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"
NUM_ACQ = 30
simulate._seed(1234)


# Make sure the GPU versions are correct by making simpler versions:
def form_cov(slc1, slc2, looks):
    numerator = take_looks(slc1 * slc2.conj(), *looks)
    amplitude1 = take_looks(slc1 * slc1.conj(), *looks)
    amplitude2 = take_looks(slc2 * slc2.conj(), *looks)
    return numerator / np.sqrt(amplitude1 * amplitude2)


def get_expected_cov(slcs, looks):
    # Manually (slowly) form the covariance matrix at all pixels
    N = slcs.shape[0]
    rows, cols = take_looks(slcs[0], *looks).shape
    out = np.zeros((rows, cols, N, N), dtype=np.complex64)
    for i in range(N):
        for j in range(i + 1, N):
            out[:, :, i, j] = form_cov(slcs[i], slcs[j], looks)
            out[:, :, j, i] = out[:, :, i, j].conj()
        out[:, :, i, i] = 1.0
    return out


@pytest.fixture(scope="module")
def slcs(shape=(10, 100, 100)):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


@pytest.fixture()
def expected_cov(slcs, looks=(5, 5)):
    return get_expected_cov(slcs, looks)


def test_coh_mat_single(slcs, expected_cov, looks=(5, 5)):
    num_slc, rows, cols = slcs.shape

    # Check the single pixel function
    expected_looked_size = tuple(
        floor(size / look) for size, look in zip((rows, cols), looks)
    )
    assert expected_cov.shape == ((*expected_looked_size, num_slc, num_slc))

    r_looks, c_looks = looks
    for r in range(expected_looked_size[0]):
        for c in range(expected_looked_size[1]):
            r_slice = slice(r * r_looks, (r + 1) * r_looks)
            c_slice = slice(c * c_looks, (c + 1) * c_looks)
            cur_samples = slcs[:, r_slice, c_slice].reshape(num_slc, -1)
            cur_C = covariance.coh_mat_single(cur_samples)
            npt.assert_array_almost_equal(expected_cov[r, c, :, :], cur_C)


def test_estimate_stack_covariance(slcs, expected_cov, looks=(5, 5)):
    # Check the full stack function
    r_looks, c_looks = looks
    half_window = HalfWindow(c_looks // 2, r_looks // 2)
    strides = Strides(c_looks, r_looks)
    C1 = covariance.estimate_stack_covariance(
        slcs, half_window=half_window, strides=strides
    )
    npt.assert_array_almost_equal(expected_cov, C1)


def test_estimate_stack_covariance_nans_pixel(slcs):
    num_slc, _, _ = slcs.shape

    C = covariance.coh_mat_single(slcs.reshape(num_slc, -1))
    # Nans for one pixel in all SLCs
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, 1, 1] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    C_nan = covariance.coh_mat_single(slc_samples_nan)
    assert np.max(np.abs(C - C_nan)) < 0.01


def test_estimate_stack_covariance_nans_image(slcs):
    num_slc, _, _ = slcs.shape
    # Nans for an entire SLC
    slc_stack_nan = slcs.copy()
    slc_stack_nan[1, :, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    # TODO: should we raise an error if we pass a dead SLC?
    assert covariance.coh_mat_single(slc_samples_nan)[0, 1] == 0
    assert covariance.coh_mat_single(slc_samples_nan)[1, 0] == 0


def test_estimate_stack_covariance_all_nans(slcs):
    num_slc, _, _ = slcs.shape
    # all nans should return an identity matrix (no correlation anywhere)
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, :, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    # TODO: do we want an identity matrix? Or for it to be 0?
    assert covariance.coh_mat_single(slc_samples_nan).sum() == 0
    # assert np.isnan(covariance.coh_mat_single(slc_samples_nan)).all()


def test_estimate_stack_covariance_neighbors(slcs):
    num_slc, rows, cols = slcs.shape

    C = covariance.coh_mat_single(slcs.reshape(num_slc, -1))
    # We're checking that setting the pixels to nan
    # is the same as saying they are not neighbors

    # Nans for one pixel in all SLCs
    slc_stack_nan = slcs.copy()
    nan_row, nan_col = (1, 1)
    slc_stack_nan[:, nan_row, nan_col] = np.nan

    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    C_nan = covariance.coh_mat_single(slc_samples_nan)

    slc_samples = slcs.reshape(num_slc, -1)
    neighbor_mask = np.ones(slc_samples.shape[1], dtype=np.bool_)
    idx = np.ravel_multi_index((nan_row, nan_col), (rows, cols))
    neighbor_mask[idx] = False

    C_neighbors = covariance.coh_mat_single(slc_samples, neighbor_mask=neighbor_mask)
    npt.assert_allclose(C_nan, C_neighbors)

    with pytest.raises(AssertionError):
        # Make sure this is different than the original
        npt.assert_allclose(C, C_neighbors)


def test_estimate_stack_covariance_neighbors_masked(slcs):
    num_slc, rows, cols = slcs.shape
    # Now mask an entire row
    slc_stack_nan = slcs.copy()
    slc_stack_nan[:, 0, :] = np.nan
    slc_samples_nan = slc_stack_nan.reshape(num_slc, -1)
    C_nan = covariance.coh_mat_single(slc_samples_nan)

    slc_samples = slcs.reshape(num_slc, -1)
    neighbor_mask = np.ones(slc_samples.shape[1], dtype=np.bool_)
    neighbor_mask[:] = True
    neighbor_mask[:cols] = False

    C_neighbors = covariance.coh_mat_single(slc_samples, neighbor_mask=neighbor_mask)
    npt.assert_allclose(C_nan, C_neighbors)
