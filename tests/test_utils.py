import datetime
from pathlib import Path

import numpy as np
import pytest

from dolphin import utils
from dolphin.io import load_gdal


def test_get_dates():
    assert ["20200303", "20210101"] == utils.get_dates("20200303_20210101.int")

    assert "20200303" == utils.get_dates("20200303.slc")[0]
    assert "20200303" == utils.get_dates(Path("20200303.slc"))[0]
    # Check that it's the filename, not the path
    assert "20200303" == utils.get_dates(Path("/usr/19990101/asdf20200303.tif"))[0]
    assert "20200303" == utils.get_dates("/usr/19990101/asdf20200303.tif")[0]

    assert ["20200303", "20210101"] == utils.get_dates(
        "/usr/19990101/20200303_20210101.int"
    )

    assert utils.get_dates("/usr/19990101/notadate.tif") == []


def test_parse_slc_strings():
    dt = datetime.date(2020, 3, 3)
    assert utils.parse_slc_strings(Path("/usr/19990101/asdf20200303.tif")) == dt
    assert utils.parse_slc_strings("/usr/19990101/asdf20200303.tif") == dt
    assert utils.parse_slc_strings("20200303.tif") == dt
    assert utils.parse_slc_strings("20200303") == dt
    assert utils.parse_slc_strings("20200303.slc") == dt

    assert utils.parse_slc_strings(["20200303.slc", "20200303.tif"]) == [dt, dt]

    with pytest.raises(ValueError):
        utils.parse_slc_strings("notadate.tif")


def test_get_types():
    np_dtype = np.dtype("complex64")
    assert 10 == utils.numpy_to_gdal_type(np_dtype)
    assert np_dtype == utils.gdal_to_numpy_type(10)

    # round trip float32
    assert utils.gdal_to_numpy_type(utils.numpy_to_gdal_type(np.float32)) == np.float32


def test_get_raster_xysize(raster_100_by_200):
    arr = load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)
    assert (200, 100) == utils.get_raster_xysize(raster_100_by_200)


def test_take_looks():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

    downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
    np.testing.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
    downsampled = utils.take_looks(arr, 2, 1, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
    downsampled = utils.take_looks(arr, 1, 2, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[0.055], [3.5]]))


def test_take_looks_3d():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])
    arr3d = np.stack([arr, arr, arr], axis=0)
    downsampled = utils.take_looks(arr3d, 2, 1)
    expected = np.array([[3.1, 4.01, 3.0 + 1.0j]])
    for i in range(3):
        np.testing.assert_array_equal(downsampled[i], expected)


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

    np.testing.assert_array_almost_equal(s1, s2, decimal=5)
