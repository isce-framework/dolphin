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


def test_take_looks():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

    downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
    npt.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
    downsampled = utils.take_looks(arr, 2, 1, func_type="mean")
    npt.assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
    downsampled = utils.take_looks(arr, 1, 2, func_type="mean")
    npt.assert_array_equal(downsampled, np.array([[0.055], [3.5]]))


def test_take_looks_3d():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])
    arr3d = np.stack([arr, arr, arr], axis=0)
    downsampled = utils.take_looks(arr3d, 2, 1)
    expected = np.array([[3.1, 4.01, 3.0 + 1.0j]])
    for i in range(3):
        npt.assert_array_equal(downsampled[i], expected)


def test_masked_looks(slc_samples):
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


def test_moving_window_mean_basic():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = utils.moving_window_mean(image, 3)
    expected = np.array(
        [
            [1.33333333, 2.33333333, 1.77777778],
            [3.0, 5.0, 3.66666667],
            [2.66666667, 4.33333333, 3.11111111],
        ]
    )
    assert np.allclose(result, expected)


def test_moving_window_mean_single_pixel():
    image = np.array([[5]])
    result = utils.moving_window_mean(image, 1)
    expected = np.array([[5]])
    assert np.allclose(result, expected)


def test_moving_window_mean_even_size():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        utils.moving_window_mean(image, (2, 2))


def test_moving_window_mean_invalid_size_type():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        utils.moving_window_mean(image, (1, 2, 3))


def test_moving_window_mean_empty_image():
    image = np.array([[]])
    result = utils.moving_window_mean(image, 1)
    expected = np.array([[]])
    assert np.allclose(result, expected)
