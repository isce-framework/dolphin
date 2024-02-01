from pathlib import Path

import pytest
from make_netcdf import create_test_nc
from pyproj import CRS

from dolphin import io, stitching
from dolphin._types import Bbox


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


@pytest.fixture()
def shifted_slc_bounds():
    return Bbox(-5.5, -4.5, 8.5, 9.5)


def test_get_combined_bounds_gt(shifted_slc_files, shifted_slc_bounds):
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
    bnds, nd = stitching.get_combined_bounds_nodata(
        shifted_slc_files[0],
    )
    expected_bnds = (-5.5, -4.5, 4.5, 5.5)
    expected_nd = None
    assert bnds == expected_bnds
    assert nd == expected_nd

    # check same file twice
    bnds, nd = stitching.get_combined_bounds_nodata(
        shifted_slc_files[0], shifted_slc_files[0]
    )
    assert bnds == expected_bnds

    # Now combined one: should have the mins as the first two values,
    # and the last two values should be the idx=-1 file
    bnds, nd = stitching.get_combined_bounds_nodata(*shifted_slc_files)
    assert bnds == shifted_slc_bounds


def test_get_combined_bounds_gt_different_proj(
    slc_file_list_nc, slc_file_list_nc_wgs84
):
    bnds, gt = stitching.get_combined_bounds_nodata(*slc_file_list_nc)
    assert bnds == (-5.5, -2.0, 4.5, 3.0)

    with pytest.raises(ValueError):
        stitching.get_combined_bounds_nodata(
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


def test_merge_images(tmp_path, shifted_slc_files, shifted_slc_bounds):
    outfile = tmp_path / "stitched.tif"
    stitching.merge_images(shifted_slc_files, outfile, target_aligned_pixels=False)

    b = io.get_raster_bounds(outfile)
    assert b == shifted_slc_bounds

    # The target aligned pixels option makes the bounds a multiple of dx/dy
    # which is 1.0/1.0 here
    outfile = tmp_path / "stitched_tap.tif"
    # Now the top/bottom must divide by 3, and left/right divide by 2
    stitching.merge_images(shifted_slc_files, outfile, target_aligned_pixels=True)

    b = io.get_raster_bounds(outfile)
    expected_bounds_tap = Bbox(-6.0, -5.0, 9.0, 10.0)
    assert b == expected_bounds_tap


def test_merge_images_strided(tmp_path, shifted_slc_files, shifted_slc_bounds):
    strides = {"x": 2, "y": 3}
    outfile = tmp_path / "stitched.tif"
    stitching.merge_images(
        shifted_slc_files,
        outfile,
        target_aligned_pixels=False,
        out_bounds=shifted_slc_bounds,
        strides=strides,
    )

    b = io.get_raster_bounds(outfile)
    assert b == shifted_slc_bounds

    # The target aligned pixels option makes the bounds a multiple of dx/dy
    # which is 1.0/1.0 here
    outfile = tmp_path / "stitched_tap.tif"
    expected_bounds_tap = Bbox(-6.0, -6.0, 10.0, 12.0)
    stitching.merge_images(
        shifted_slc_files, outfile, target_aligned_pixels=True, strides=strides
    )

    b = io.get_raster_bounds(outfile)
    assert b == expected_bounds_tap


@pytest.mark.parametrize("buffer", [0, 1.5])
@pytest.mark.parametrize("strides", [{"x": 1, "y": 1}, {"x": 3, "y": 2}])
def test_merge_images_specify_bounds(
    tmp_path, strides, buffer, shifted_slc_files, shifted_slc_bounds
):
    from shapely.geometry import box

    outfile = tmp_path / "stitched.tif"

    buffered_box = box(*shifted_slc_bounds).buffer(buffer)
    buffered_bounds = buffered_box.bounds
    stitching.merge_images(
        shifted_slc_files,
        outfile,
        target_aligned_pixels=False,
        out_bounds=buffered_bounds,
        strides=strides,
    )

    b = io.get_raster_bounds(outfile)
    assert b == buffered_bounds


def test_merge_images_one_image_out_bounds_specified(
    tmp_path, shifted_slc_files, shifted_slc_bounds
):
    from shapely.geometry import box

    outfile = tmp_path / "stitched.tif"

    buffered_box = box(*shifted_slc_bounds).buffer(1.5)
    buffered_bounds = buffered_box.bounds

    # only grab the first image
    # without specifying bounds, this should not work
    stitching.merge_images(shifted_slc_files[:1], outfile)

    b = io.get_raster_bounds(outfile)
    assert b != buffered_bounds

    outfile2 = tmp_path / "stitched_with_bounds.tif"
    # after specifying bounds, this should work
    stitching.merge_images(
        shifted_slc_files[:1],
        outfile2,
        target_aligned_pixels=False,
        out_bounds=buffered_bounds,
    )

    b = io.get_raster_bounds(outfile2)
    assert b == buffered_bounds
