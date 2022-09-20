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
