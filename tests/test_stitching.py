from pathlib import Path

import pytest
from make_netcdf import create_test_nc

from dolphin import io, stitching


@pytest.fixture()
def shifted_slc_files(tmp_path):
    """Make series of files offset in lat/lon."""
    shape = (5, 10, 10)

    start_date = 20220101
    name_template = tmp_path / "shifted_{date}.nc"
    file_list = []
    for i in range(shape[0]):
        fname = str(name_template).format(date=str(start_date + i))
        create_test_nc(fname, epsg=4326, subdir="/", shape=shape[1:], xoff=i, yoff=i)
        file_list.append(Path(fname))

    return file_list


def test_get_combined_bounds(shifted_slc_files):
    # Use the created WGS84 SLC
    bnds = io.get_raster_bounds(shifted_slc_files[0])
    expected = (-5.5, -4.5, 4.5, 5.5)
    assert bnds == expected
    bnds = io.get_raster_bounds(shifted_slc_files[1])
    expected = (-4.5, -3.5, 5.5, 6.5)
    assert bnds == expected
    bnds = io.get_raster_bounds(shifted_slc_files[-1])
    expected = (-1.5, -0.5, 8.5, 9.5)
    assert bnds == expected

    # check same file twice
    bnds = stitching.get_combined_bounds(shifted_slc_files[0], shifted_slc_files[0])
    expected = (-5.5, -4.5, 4.5, 5.5)
    assert bnds == expected

    # Now combined one: should have the mins as the first two values,
    # and the last two values should be the -1 file
    bnds = stitching.get_combined_bounds(*shifted_slc_files)
    expected = (-5.5, -4.5, 8.5, 9.5)
    assert bnds == expected
