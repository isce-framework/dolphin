import itertools
from datetime import datetime, timedelta

import numpy as np
import numpy.testing as npt
import pytest
from numpy.linalg import lstsq as lstsq_numpy

from dolphin import io, timeseries
from dolphin.utils import format_dates

NUM_DATES = 10
DT = 12
START_DATE = datetime(2020, 1, 1)
SHAPE = 50, 50
VELO_RAD_PER_YEAR = 5.0
VELO_RAD_PER_DAY = VELO_RAD_PER_YEAR / 365.25


def make_sar_dates():
    sar_date_list = []
    for idx in range(NUM_DATES):
        sar_date_list.append(START_DATE + timedelta(days=idx * DT))
    return sar_date_list


def make_sar_phases(sar_dates):
    out = np.empty((NUM_DATES, *SHAPE), dtype="float32")
    # Make a linear ramp which increases each time interval
    ramp0 = np.ones((SHAPE[0], 1)) * np.arange(SHAPE[1]).reshape(1, -1)
    ramp0 /= SHAPE[1]
    for idx in range(len(sar_dates)):
        cur_slope = VELO_RAD_PER_DAY * DT * idx
        out[idx] = cur_slope * ramp0
    return out


def make_ifg_date_pairs(sar_dates):
    """Form all possible unwrapped interferogram pairs from the date list"""
    return [tuple(pair) for pair in itertools.combinations(sar_dates, 2)]


def make_ifgs(sar_phases):
    """Form all possible unwrapped interferogram pairs from the sar images"""
    return np.stack(
        [(sec - ref) for (ref, sec) in itertools.combinations(sar_phases, 2)]
    )


@pytest.fixture(scope="module")
def data():
    sar_dates = make_sar_dates()
    sar_phases = make_sar_phases(sar_dates)
    ifg_date_pairs = make_ifg_date_pairs(sar_dates)
    ifgs = make_ifgs(sar_phases)
    return sar_dates, sar_phases, ifg_date_pairs, ifgs


def solve_with_removal(A, b):
    good_idxs = (~np.isnan(b)).flatten()
    b_clean = b[good_idxs]
    A_clean = A[good_idxs, :]
    return lstsq_numpy(A_clean, b_clean)[0]


def solve_by_zeroing(A, b):
    weight = (~np.isnan(b)).astype(float)
    A0 = A.copy()
    A0 = A * weight[:, None]
    b2 = np.nan_to_num(b)
    return lstsq_numpy(A0, b2)[0]


class TestUtils:
    def test_datetime_to_float(self):
        sar_dates = make_sar_dates()
        date_arr = timeseries.datetime_to_float(sar_dates)
        expected = np.array(
            [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0]
        )
        assert np.allclose(date_arr, expected)

    def test_incidence_matrix(self):
        A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        assert A.shape == (5, 5)
        expected = np.array(
            [
                [1, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0],
                [0, -1, 1, 0, 0],
                [0, 0, -1, 1, 0],
                [0, 0, 0, -1, 1],
            ]
        )
        np.testing.assert_array_equal(A, expected)


class TestInvert:
    @pytest.fixture
    def A(self, data):
        _, _, ifg_date_pairs, _ = data
        return timeseries.get_incidence_matrix(ifg_date_pairs)

    def test_basic(self, data, A):
        sar_dates, sar_phases, _ifg_date_pairs, ifgs = data

        # Get one pixel
        b = ifgs[:, -1, -1]
        weights = np.ones_like(b)
        x, residual = timeseries.weighted_lstsq_single(A, b, weights)
        assert x.shape[0] == len(sar_dates) - 1
        npt.assert_allclose(x, sar_phases[1:, -1, -1], atol=1e-5)
        assert np.array(residual).item() < 1e-5

    def test_l1_invert(self, data, A):
        sar_dates, sar_phases, _ifg_date_pairs, ifgs = data

        # Get one pixel
        b = ifgs[:, -1, -1].copy()
        # Change one value by a lot, should ignore it
        single_ifg_error = 100
        b[0] += single_ifg_error

        R = np.linalg.cholesky(A.T @ A)
        x, _residual = timeseries.least_absolute_deviations(A, b, R)
        residuals = np.abs(b - A @ x)
        assert x.shape[0] == len(sar_dates) - 1
        npt.assert_allclose(x, sar_phases[1:, -1, -1], atol=1e-4)
        npt.assert_allclose(residuals[0], single_ifg_error, atol=1e-4)
        npt.assert_allclose(residuals[1:], 0, atol=1e-4)

    def test_invert_stack(self, data, A):
        _sar_dates, sar_phases, _ifg_date_pairs, ifgs = data

        phi_stack, residuals = timeseries.invert_stack(A, ifgs)
        assert phi_stack.shape == sar_phases[1:].shape
        assert residuals.shape == sar_phases[0].shape
        npt.assert_allclose(phi_stack, sar_phases[1:], atol=1e-5)
        npt.assert_array_less(residuals, 1e-5)

    def test_invert_stack_l1(self, data, A):
        _sar_dates, sar_phases, _ifg_date_pairs, ifgs = data
        ifgs = ifgs.copy()
        single_ifg_error = 100
        ifgs[0] += single_ifg_error

        phi_stack, residuals = timeseries.invert_stack_l1(A, ifgs)
        assert phi_stack.shape == sar_phases[1:].shape
        assert residuals.shape == sar_phases[0].shape
        npt.assert_allclose(phi_stack, sar_phases[1:], atol=1e-3)
        npt.assert_allclose(residuals, single_ifg_error, atol=1e-3)

    def test_weighted_stack(self, data, A):
        sar_dates, sar_phases, _ifg_date_pairs, ifgs = data

        weights = np.ones_like(ifgs)
        phi, residuals = timeseries.invert_stack(A, ifgs, weights)
        assert phi.shape[0] == len(sar_dates) - 1
        npt.assert_allclose(phi, sar_phases[1:], atol=1e-5)
        # Here there is no noise, so it's over determined
        weights[1, :, :] = 0
        phi2, residuals2 = timeseries.invert_stack(A, ifgs, weights)
        npt.assert_allclose(phi2, sar_phases[1:], atol=1e-5)
        npt.assert_allclose(residuals2, residuals, atol=1e-5)

    def test_censored_stack(self, data, A):
        sar_dates, sar_phases, _ifg_date_pairs, ifgs = data

        weights = None
        missing_data_flags = np.ones(ifgs.shape, dtype=bool)  # no missing data
        phi, residuals = timeseries.invert_stack(A, ifgs, weights, missing_data_flags)
        assert phi.shape[0] == len(sar_dates) - 1
        npt.assert_allclose(phi, sar_phases[1:], atol=1e-5)

        # Here there is no noise, so it's over determined
        missing_data_flags = np.random.rand(*ifgs.shape) > 0.05  # 5% missing
        phi2, residuals2 = timeseries.invert_stack(A, ifgs, weights, missing_data_flags)
        npt.assert_allclose(phi2, sar_phases[1:], atol=1e-5)
        npt.assert_allclose(residuals2, residuals, atol=1e-5)

    def test_remove_row_vs_weighted(self, data, A):
        """Check that removing a row/data point is equivalent to zero-weighting it."""
        _sar_dates, _sar_phases, _ifg_date_pairs, ifgs = data
        dphi = ifgs[:, -1, -1]
        # add noise to the ifgs, or we can't tell a difference
        dphi_noisy = dphi + np.random.normal(0, 0.3, dphi.shape)

        # Once with a zeroed weight
        weights = np.ones_like(dphi_noisy)
        weights[1] = 0
        phi_0, residual_0 = timeseries.weighted_lstsq_single(A, dphi_noisy, weights)

        A2 = A.copy()
        # delete a row
        A2 = np.delete(A2, 1, axis=0)
        # delete the data point
        dphi2 = np.delete(dphi_noisy, 1)
        weights2 = np.delete(weights, 1)
        phi2, residual2 = timeseries.weighted_lstsq_single(A2, dphi2, weights2)
        npt.assert_allclose(phi_0, phi2, atol=1e-5)

        npt.assert_allclose(residual_0, residual2, atol=1e-5)

    @pytest.fixture
    def unw_files(self, tmp_path, data, raster_100_by_200):
        """Write the data to disk and return the file names."""
        _sar_dates, _sar_phases, ifg_date_pairs, ifgs = data

        out = []
        # use the raster as a template
        for pair, ifg in zip(ifg_date_pairs, ifgs, strict=False):
            fname = tmp_path / (format_dates(*pair) + ".tif")
            io.write_arr(arr=ifg, output_name=fname, like_filename=raster_100_by_200)
            out.append(fname)

        return out

    @pytest.mark.parametrize("method", ["L1", "L2"])
    def test_invert_unw_network(self, data, unw_files, tmp_path, method):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ref_point = (0, 0)
        out_files, out_residuals = timeseries.invert_unw_network(
            unw_file_list=unw_files,
            reference=ref_point,
            output_dir=output_dir,
            method=method,
            # ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
            # block_shape: tuple[int, int] = (512, 512),
            # cor_file_list: Sequence[PathOrStr] | None = None,
            # cor_threshold: float = 0.2,
            num_threads=1,
        )
        # Check results
        solved_stack = io.RasterStackReader.from_file_list(out_files)[:, :, :]
        sar_phases = data[1]
        # Account for the flip in sign for LOS convention:
        npt.assert_allclose(solved_stack, -1 * sar_phases[1:], atol=1e-5)

        residuals = io.RasterStackReader.from_file_list(out_residuals)[:, :, :]
        npt.assert_allclose(residuals, 0, atol=1e-4)


class TestVelocity:
    @pytest.fixture
    def x_arr(self, data):
        sar_dates, _sar_phases, _ifg_date_pairs, _ifgs = data
        return timeseries.datetime_to_float(sar_dates)

    @pytest.fixture
    def expected_velo(self, data, x_arr):
        _sar_dates, sar_phases, _ifg_date_pairs, _ifgs = data
        out = np.zeros(SHAPE)
        for i in range(SHAPE[0]):
            for j in range(SHAPE[1]):
                out[i, j] = np.polyfit(x_arr, sar_phases[:, i, j], 1)[0]
        return out * 365.25

    def test_stack(self, data, x_arr, expected_velo):
        _sar_dates, sar_phases, _ifg_date_pairs, _ifgs = data

        weights = np.ones_like(sar_phases)
        velocities = timeseries.estimate_velocity(x_arr, sar_phases, weights)
        assert velocities.shape == (SHAPE[0], SHAPE[1])
        npt.assert_allclose(velocities, expected_velo, atol=1e-5)

    def test_stack_unweighted(self, data, x_arr, expected_velo):
        _sar_dates, sar_phases, _ifg_date_pairs, _ifgs = data

        velocities = timeseries.estimate_velocity(x_arr, sar_phases, None)
        assert velocities.shape == (SHAPE[0], SHAPE[1])
        npt.assert_allclose(velocities, expected_velo, atol=1e-5)


class TestReferencePoint:
    def test_all_ones_center(self, tmp_path):
        # All coherence=1 => pick the true center pixel.
        shape = (31, 31)
        arr = np.ones(shape, dtype="float32")

        coh_file = tmp_path / "coh_ones.tif"
        io.write_arr(arr=arr, output_name=coh_file)

        ref_point = timeseries.select_reference_point(
            quality_file=coh_file,
            output_dir=tmp_path,
            candidate_threshold=0.95,  # everything is above 0.95
            ccl_file_list=None,
        )
        npt.assert_equal((ref_point.row, ref_point.col), (15, 15))

    def test_half_03_half_099(self, tmp_path):
        # Left half=0.3, right half=0.99 => reference point should be
        # near the center of the right side.
        shape = (31, 31)
        arr = np.full(shape, 0.3, dtype="float32")
        # Make right half, 16 to 30, high coherence
        arr[:, 16:] = 0.99

        coh_file = tmp_path / "coh_half_03_099.tif"
        io.write_arr(arr=arr, output_name=coh_file)

        ref_point = timeseries.select_reference_point(
            quality_file=coh_file,
            output_dir=tmp_path,
            candidate_threshold=0.95,
            ccl_file_list=None,
        )
        # Expect the center row=15, and roughly col=23 for columns 16..30.
        npt.assert_equal((ref_point.row, ref_point.col), (15, 23))

    def test_with_conncomp(self, tmp_path):
        """Make a temporal coherence with left half=0.3, right half=1.0.

        Make the connected-component labels have top half=0, bottom half=1.

        Reference point is in the bottom-right quadrant where
        coherence > threshold AND conncomp == 1
        """
        shape = (31, 31)

        # Coherence array: left half=0.3, right half=1.0
        coh = np.full(shape, 0.3, dtype="float32")
        coh[:, 16:] = 1.0

        coh_file = tmp_path / "coh_left03_right1.tif"
        io.write_arr(arr=coh, output_name=coh_file)

        # ConnComp label: top half=0, bottom half=1
        # So only rows >= 15 are labeled '1'.
        ccl = np.zeros(shape, dtype="uint16")
        ccl[16:, :] = 1

        ccl_file1 = tmp_path / "conncomp_bottom_half.tif"
        io.write_arr(arr=ccl, output_name=ccl_file1)
        # Add another conncomp file with all good pixels
        ccl_file2 = tmp_path / "conncomp_full.tif"
        io.write_arr(arr=np.ones_like(ccl), output_name=ccl_file2)

        ref_point = timeseries.select_reference_point(
            ccl_file_list=[ccl_file1, ccl_file2],
            quality_file=coh_file,
            output_dir=tmp_path,
            candidate_threshold=0.95,
        )

        # Bottom half => rows [16..30], right half => cols [16..30].
        npt.assert_equal((ref_point.row, ref_point.col), (23, 23))


if __name__ == "__main__":
    sar_dates = make_sar_dates()
    sar_phases = make_sar_phases(sar_dates)
    ifg_date_pairs = make_ifg_date_pairs(sar_dates)
    ifgs = make_ifgs(sar_phases)
