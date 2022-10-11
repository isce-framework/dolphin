import numpy as np
from pathlib import Path

import pytest

from dolphin import utils


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

    with pytest.raises(ValueError):
        utils.get_dates("/usr/19990101/notadate.tif")


def test_get_raster_xysize(tmp_path):
    from osgeo import gdal

    xsize, ysize = 10, 20
    # Create a test raster
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(tmp_path / "test.bin"), xsize, ysize, 1, gdal.GDT_Float32)
    ds = None  # noqa

    assert (xsize, ysize) == utils.get_raster_xysize(tmp_path / "test.bin")


def test_take_looks():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

    downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
    np.testing.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
    downsampled = utils.take_looks(arr, 2, 1, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
    downsampled = utils.take_looks(arr, 1, 2, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[0.055], [3.5]]))


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