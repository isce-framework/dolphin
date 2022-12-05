import datetime
from pathlib import Path

import numpy as np
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
