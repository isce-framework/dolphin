from pathlib import Path

import pytest
from make_netcdf import create_test_nc
from pyproj import CRS

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


def test_get_combined_bounds_gt(shifted_slc_files):
    # Use the created WGS84 SLC
    bnds = io.get_raster_bounds(shifted_slc_files[0])
    expected_bnds = (-5.5, -4.5, 4.5, 5.5)
    assert bnds == expected_bnds

    bnds = io.get_raster_bounds(shifted_slc_files[1])
    expected_bnds = (-4.5, -3.5, 5.5, 6.5)
    assert bnds == expected_bnds

    bnds = io.get_raster_bounds(shifted_slc_files[-1])
    expected_bnds = (-1.5, -0.5, 8.5, 9.5)
    assert bnds == expected_bnds

    # should be same as test #1 above
    bnds, gt = stitching._get_combined_bounds_gt(
        shifted_slc_files[0],
    )
    expected_bnds = (-5.5, -4.5, 4.5, 5.5)
    expected_gt = [-5.5, 1.0, 0.0, 5.5, 0.0, -1.0]
    assert bnds == expected_bnds
    assert gt == expected_gt

    # check same file twice
    bnds, gt = stitching._get_combined_bounds_gt(
        shifted_slc_files[0], shifted_slc_files[0]
    )
    assert bnds == expected_bnds
    assert gt == expected_gt

    # Now combined one: should have the mins as the first two values,
    # and the last two values should be the -1 file
    bnds, gt = stitching._get_combined_bounds_gt(*shifted_slc_files)
    expected_bnds = (-5.5, -4.5, 8.5, 9.5)
    # only the top left corner should change
    expected_gt[0], expected_gt[3] = -5.5, 9.5
    assert bnds == expected_bnds
    assert gt == expected_gt


def test_get_combined_bounds_gt_different_proj(
    slc_file_list_nc, slc_file_list_nc_wgs84
):
    bnds, gt = stitching._get_combined_bounds_gt(*slc_file_list_nc)
    assert bnds == (-5.5, -2.0, 4.5, 3.0)
    assert gt == [-5.5, 1.0, 0, 3.0, 0, -1.0]

    with pytest.raises(ValueError):
        stitching._get_combined_bounds_gt(
            slc_file_list_nc_wgs84[0], slc_file_list_nc[0]
        )


def test_get_mode_projection(slc_file_list_nc, slc_file_list_nc_wgs84):
    p = stitching._get_mode_projection(slc_file_list_nc_wgs84[0:1])

    epsg4326 = CRS.from_epsg(4326)
    assert CRS.from_user_input(p) == epsg4326

    p = stitching._get_mode_projection(
        slc_file_list_nc[0:1],
    )
    epsg32615 = CRS.from_epsg(32615)
    assert CRS.from_user_input(p) == epsg32615

    p = stitching._get_mode_projection(
        slc_file_list_nc[0:2] + slc_file_list_nc_wgs84[0:1],
    )
    assert CRS.from_user_input(p) == epsg32615

    p = stitching._get_mode_projection(
        slc_file_list_nc[0:1] + slc_file_list_nc_wgs84[0:2],
    )
    assert CRS.from_user_input(p) == epsg4326
