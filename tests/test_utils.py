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
    assert utils.get_dates("20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert utils.get_dates("20200303.slc")[0] == datetime.date(2020, 3, 3)
    assert utils.get_dates(Path("20200303.slc"))[0] == datetime.date(2020, 3, 3)
    # Check that it's the filename, not the path
    assert utils.get_dates(Path("/usr/19990101/asdf20200303.tif"))[0] == datetime.date(
        2020, 3, 3
    )
    assert utils.get_dates("/usr/19990101/asdf20200303.tif")[0] == datetime.date(
        2020, 3, 3
    )

    assert utils.get_dates("/usr/19990101/20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert utils.get_dates("/usr/19990101/notadate.tif") == []


def test_get_dates_with_format():
    # try other date formats
    fmt = "%Y-%m-%d"
    assert utils.get_dates("2020-03-03_2021-01-01.int", fmt) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]


def test_get_dates_with_gdal_string():
    # Checks that is can parse 'NETCDF:"/path/to/file.nc":variable'
    assert utils.get_dates('NETCDF:"/usr/19990101/20200303_20210101.nc":variable') == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    assert utils.get_dates(
        'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable/2"'
    ) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    # Check the derived dataset name too
    assert utils.get_dates(
        'DERIVED_SUBDATASET:AMPLITUDE:"/usr/19990101/20200303_20210101.int"'
    ) == [datetime.date(2020, 3, 3), datetime.date(2021, 1, 1)]


def test_sort_files_by_date():
    files = [
        "slc_20200303.tif",
        "slc_20210101.tif",
        "slc_20190101.tif",
        "slc_20180101.tif",
    ]
    expected_dates = [
        [datetime.date(2018, 1, 1)],
        [datetime.date(2019, 1, 1)],
        [datetime.date(2020, 3, 3)],
        [datetime.date(2021, 1, 1)],
    ]
    expected_files = sorted(files)

    sorted_files, sorted_dates = utils.sort_files_by_date(files)
    assert sorted_files == expected_files
    assert sorted_dates == expected_dates

    # Check that it works with Path objects
    files = [Path(f) for f in files]
    sorted_files, sorted_dates = utils.sort_files_by_date(files)
    assert [Path(f) for f in expected_files] == sorted_files
    assert sorted_dates == expected_dates

    # check it ignores paths leading up to file name
    files = [
        "/usr/20200101/asdf20180101.tif",
        "/usr/19900101/asdf20190101.tif",
        "/usr/20000101/asdf20200303.tif",
        "/usr/19990101/asdf20210101.tif",
    ]
    sorted_files, sorted_dates = utils.sort_files_by_date(files)
    assert sorted_files == files  # they were in sorted order already
    assert sorted_dates == expected_dates


def test_sort_files_by_date_interferograms():
    # Make files with 2-date names
    files = [
        "ifg_20200303_20210101.tif",
        "ifg_20200303_20220101.tif",
        "ifg_20190101_20200303.tif",
        "ifg_20180101_20210101.tif",
    ]
    dates = [
        [datetime.date(2018, 1, 1), datetime.date(2021, 1, 1)],
        [datetime.date(2019, 1, 1), datetime.date(2020, 3, 3)],
        [datetime.date(2020, 3, 3), datetime.date(2021, 1, 1)],
        [datetime.date(2020, 3, 3), datetime.date(2022, 1, 1)],
    ]
    sorted_files, sorted_dates = utils.sort_files_by_date(files)
    assert sorted_files == sorted(files)  # here lexicographic order is correct
    assert sorted_dates == sorted(dates)


def test_sort_files_by_date_compressed_first():
    # Check that compressed SLCs go first, then SLCs are sorted by date
    unsorted_files = [
        "slc_20200101.tif",
        "slc_20210101.tif",
        "slc_20190101.tif",
        "compressed_20180101_20200101.tif",
        "slc_20180101.tif",
        "compressed_20200101_20210101.tif",
    ]
    expected_dates = [
        [datetime.date(2018, 1, 1), datetime.date(2020, 1, 1)],
        [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)],
        [datetime.date(2018, 1, 1)],
        [datetime.date(2019, 1, 1)],
        [datetime.date(2020, 1, 1)],
        [datetime.date(2021, 1, 1)],
    ]

    sorted_files, sorted_dates = utils.sort_files_by_date(unsorted_files)
    assert sorted_files == [
        "compressed_20180101_20200101.tif",
        "compressed_20200101_20210101.tif",
        "slc_20180101.tif",
        "slc_20190101.tif",
        "slc_20200101.tif",
        "slc_20210101.tif",
    ]
    assert sorted_dates == expected_dates


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
