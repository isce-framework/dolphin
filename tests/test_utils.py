import datetime
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin import utils


def test_get_types():
    np_dtype = np.dtype("complex64")
    assert 10 == utils.numpy_to_gdal_type(np_dtype)
    assert np_dtype == utils.gdal_to_numpy_type(10)

    # round trip float32
    assert utils.gdal_to_numpy_type(utils.numpy_to_gdal_type(np.float32)) == np.float32


def test_date_format_to_regex():
    # Test date format strings with different specifiers and delimiters
    matching_dates = [
        ("%Y-%m-%d", "2021-01-01"),
        ("%Y/%m/%d", "2022/02/02"),
        ("%Y%m%d", "20230103"),
        ("%d-%m-%Y", "01-04-2024"),
        ("%m/%d/%Y", "05/06/2025"),
    ]
    for date_format, date in matching_dates:
        pattern = utils._date_format_to_regex(date_format)

        # Test that the date matches the regular expression
        assert pattern.match(date) is not None

    # Test date formats that should not match the dates in "non_matching_dates"
    date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y"]
    non_matching_dates = ["01-01-2021", "2022-02-03", "2022-03-04", "2022/05/06"]
    for date, date_format in zip(non_matching_dates, date_formats):
        pattern = utils._date_format_to_regex(date_format)

        # Test that the date does not match the regular expression
        assert pattern.match(date) is None


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

    # try other date formats
    fmt = "%Y-%m-%d"
    assert ["2020-03-03", "2021-01-01"] == utils.get_dates(
        "2020-03-03_2021-01-01.int", fmt
    )


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

    fmt = "%Y-%m-%d"
    assert utils.parse_slc_strings(["2020-03-03.slc", "2020-03-03.tif"], fmt=fmt) == [
        dt,
        dt,
    ]


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
