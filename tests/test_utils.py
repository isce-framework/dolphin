import datetime
from pathlib import Path

import numpy as np

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


def test_sort_files_by_date():
    files = [
        "slc_20200303.tif",
        "slc_20210101.tif",
        "slc_20190101.tif",
        "slc_20180101.tif",
    ]
    expected_dates = [
        datetime.date(2018, 1, 1),
        datetime.date(2019, 1, 1),
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
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
