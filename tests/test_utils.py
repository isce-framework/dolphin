import datetime
from pathlib import Path

import numpy as np
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
