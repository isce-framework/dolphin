import numpy as np
import numpy.testing as npt
import pytest

from dolphin import utils


def test_get_types():
    np_dtype = np.dtype("complex64")
    assert utils.numpy_to_gdal_type(np_dtype) == 10
    assert np_dtype == utils.gdal_to_numpy_type(10)

    # round trip float32
    assert utils.gdal_to_numpy_type(utils.numpy_to_gdal_type(np.float32)) == np.float32


class TestTakeLooks:
    def test_basic(self):
        arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

        downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
        npt.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
        downsampled = utils.take_looks(arr, 2, 1, func_type="mean")
        npt.assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
        downsampled = utils.take_looks(arr, 1, 2, func_type="mean")
        npt.assert_array_equal(downsampled, np.array([[0.055], [3.5]]))

    def test_3d(self):
        arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])
        arr3d = np.stack([arr, arr, arr], axis=0)
        downsampled = utils.take_looks(arr3d, 2, 1)
        expected = np.array([[3.1, 4.01, 3.0 + 1.0j]])
        for i in range(3):
            npt.assert_array_equal(downsampled[i], expected)

    def test_nans(self, slc_samples):
        slc_stack = slc_samples.reshape(30, 11, 11)
        mask = np.zeros((11, 11), dtype=bool)
        # Mask the top row
        mask[0, :] = True
        slc_samples_masked = slc_stack[:, ~mask]
        s1 = np.nansum(slc_samples_masked, axis=1)

        slc_stack_masked = slc_stack.copy()
        slc_stack_masked[:, mask] = np.nan
        s2 = np.squeeze(utils.take_looks(slc_stack_masked, 11, 11))

        npt.assert_array_almost_equal(s1, s2, decimal=5)

    def test_masked_array(self):
        arr = np.ma.MaskedArray(
            [[-999, 3, 4, 1 + 1j]], mask=[[True, False, False, False]]
        )
        downsampled = utils.take_looks(arr, 1, 2, func_type="nansum")
        npt.assert_array_equal(
            downsampled, np.ma.MaskedArray([[np.inf, 5 + 1j]], mask=[True, False])
        )
        assert type(downsampled) is np.ma.MaskedArray

    def test_nomask(self):
        arr = np.ma.MaskedArray([[0.1, 0.01, 2], [3, 4, 1 + 1j]], mask=np.ma.nomask)
        downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
        npt.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
        assert type(downsampled) is np.ma.MaskedArray


def test_upsample_nearest():
    arr = np.arange(16).reshape(4, 4)
    looked = utils.take_looks(arr, 2, 2, func_type="max")
    assert looked.shape == (2, 2)
    npt.assert_array_equal(looked, np.array([[5, 7], [13, 15]]))

    upsampled = utils.upsample_nearest(looked, output_shape=arr.shape)
    assert upsampled.shape == (4, 4)
    npt.assert_array_equal(
        upsampled,
        np.array(
            [
                [5, 5, 7, 7],
                [5, 5, 7, 7],
                [13, 13, 15, 15],
                [13, 13, 15, 15],
            ]
        ),
    )

    arr3d = np.stack([arr, arr, arr], axis=0)
    looked3d = utils.take_looks(arr3d, 2, 2, func_type="max")
    assert looked3d.shape == (3, 2, 2)
    upsampled3d = utils.upsample_nearest(looked3d, output_shape=arr.shape)
    assert upsampled3d.shape == (3, 4, 4)
    for img in upsampled3d:
        npt.assert_array_equal(img, upsampled)


class TestGrowNodataRegion:
    @pytest.mark.parametrize("nodata", [0, -9999.0, np.nan])
    def test_basic_rectangle(self, nodata):
        """Test basic rectangular data with nodata border."""
        arr = np.full((5, 5), nodata)
        # rows/cols 1,2,3 are valid
        arr[1:4, 1:4] = 1
        # After eroding 1, only the middle is left
        expected = np.full((5, 5), nodata)
        expected[2, 2] = 1

        result = utils.grow_nodata_region(arr, nodata=nodata, n_pixels=1)
        npt.assert_array_equal(result, expected)

    def test_diagonal(self):
        """Test diagonal line of data."""
        arr = np.full((6, 6), -9999.0)
        # Create diagonal line of data
        np.fill_diagonal(arr, 100)
        result = utils.grow_nodata_region(arr, nodata=-9999.0, n_pixels=1)

        # Everything should be eroded due to 9-connectivity
        expected = np.full((6, 6), -9999.0)
        npt.assert_array_equal(result, expected)

    def test_irregular_shape(self):
        """Test irregular L-shaped data pattern."""
        arr = np.full((5, 5), 0)
        # Create L-shaped pattern
        arr[1:4, 1] = 1  # vertical line
        arr[3, 1:4] = 1  # horizontal line

        result = utils.grow_nodata_region(arr, nodata=0, n_pixels=1)
        # After erosion, both single-pixel lines are gone
        expected = np.full((5, 5), 0)
        npt.assert_array_equal(result, expected)

    def test_no_copy(self):
        """Test in-place modification when copy=False."""
        arr = np.full((4, 4), 0)
        arr[1:3, 1:3] = 1
        original_arr = arr.copy()

        result = utils.grow_nodata_region(arr, nodata=0, n_pixels=1, copy=False)
        assert result is arr  # Should return same object
        assert not np.array_equal(arr, original_arr)  # Should be modified

    def test_larger_window(self):
        """Test erosion with larger window size."""
        arr = np.full((7, 7), 0)
        arr[1:6, 1:6] = 1

        result = utils.grow_nodata_region(arr, nodata=0, n_pixels=2)
        expected = np.full((7, 7), 0)
        expected[3, 3] = 1
        npt.assert_array_equal(result, expected)

    def test_all_nodata(self):
        """Test array with all nodata values."""
        arr = np.full((4, 4), 0)
        result = utils.grow_nodata_region(arr, nodata=0, n_pixels=1)
        npt.assert_array_equal(result, arr)

    def test_all_valid(self):
        """Test array with all valid values."""
        arr = np.full((4, 4), 1)
        result = utils.grow_nodata_region(arr, nodata=0, n_pixels=1)
        npt.assert_array_equal(result, arr)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            np.array([1, 2, 3]),  # 1D array
            np.array([[[1]]]),  # 3D array
        ],
    )
    def test_invalid_dimensions(self, invalid_input):
        """Test that function raises error for non-2D arrays."""
        with pytest.raises(ValueError):
            utils.grow_nodata_region(invalid_input, nodata=-9999.0)
