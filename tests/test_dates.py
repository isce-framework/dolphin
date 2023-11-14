import datetime
from pathlib import Path

from dolphin import _dates


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
        pattern = _dates._date_format_to_regex(date_format)

        # Test that the date matches the regular expression
        assert pattern.match(date) is not None

    # Test date formats that should not match the dates in "non_matching_dates"
    date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y"]
    non_matching_dates = ["01-01-2021", "2022-02-03", "2022-03-04", "2022/05/06"]
    for date, date_format in zip(non_matching_dates, date_formats):
        pattern = _dates._date_format_to_regex(date_format)

        # Test that the date does not match the regular expression
        assert pattern.match(date) is None


def test_get_dates():
    assert _dates.get_dates("20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert _dates.get_dates("20200303.slc")[0] == datetime.date(2020, 3, 3)
    assert _dates.get_dates(Path("20200303.slc"))[0] == datetime.date(2020, 3, 3)
    # Check that it's the filename, not the path
    assert _dates.get_dates(Path("/usr/19990101/asdf20200303.tif"))[0] == datetime.date(
        2020, 3, 3
    )
    assert _dates.get_dates("/usr/19990101/asdf20200303.tif")[0] == datetime.date(
        2020, 3, 3
    )

    assert _dates.get_dates("/usr/19990101/20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert _dates.get_dates("/usr/19990101/notadate.tif") == []


def test_get_dates_with_format():
    # try other date formats
    fmt = "%Y-%m-%d"
    assert _dates.get_dates("2020-03-03_2021-01-01.int", fmt) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]


def test_get_dates_with_gdal_string():
    # Checks that is can parse 'NETCDF:"/path/to/file.nc":variable'
    assert _dates.get_dates('NETCDF:"/usr/19990101/20200303_20210101.nc":variable') == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    assert _dates.get_dates(
        'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable/2"'
    ) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    # Check the derived dataset name too
    assert _dates.get_dates(
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

    sorted_files, sorted_dates = _dates.sort_files_by_date(files)
    assert sorted_files == expected_files
    assert sorted_dates == expected_dates

    # Check that it works with Path objects
    files = [Path(f) for f in files]
    sorted_files, sorted_dates = _dates.sort_files_by_date(files)
    assert [Path(f) for f in expected_files] == sorted_files
    assert sorted_dates == expected_dates

    # check it ignores paths leading up to file name
    files = [
        "/usr/20200101/asdf20180101.tif",
        "/usr/19900101/asdf20190101.tif",
        "/usr/20000101/asdf20200303.tif",
        "/usr/19990101/asdf20210101.tif",
    ]
    sorted_files, sorted_dates = _dates.sort_files_by_date(files)
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
    sorted_files, sorted_dates = _dates.sort_files_by_date(files)
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

    sorted_files, sorted_dates = _dates.sort_files_by_date(unsorted_files)
    assert sorted_files == [
        "compressed_20180101_20200101.tif",
        "compressed_20200101_20210101.tif",
        "slc_20180101.tif",
        "slc_20190101.tif",
        "slc_20200101.tif",
        "slc_20210101.tif",
    ]
    assert sorted_dates == expected_dates
