from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

from dolphin.stack import VRTStack
from dolphin.utils import _get_path_from_gdal_str

# Note: uses the fixtures from conftest.py


@pytest.fixture
def vrt_stack(tmp_path, slc_stack, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    assert len(s) == len(slc_stack) == len(slc_file_list)
    return s


@pytest.fixture
def vrt_stack_nc(tmp_path, slc_stack, slc_file_list_nc):
    vrt_file = tmp_path / "test_nc.vrt"
    s = VRTStack(slc_file_list_nc, outfile=vrt_file, subdataset="data")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture
def vrt_stack_nc_subdataset(tmp_path, slc_stack, slc_file_list_nc_with_sds):
    vrt_file = tmp_path / "test_nc.vrt"
    files_only = [_get_path_from_gdal_str(f) for f in slc_file_list_nc_with_sds]
    s = VRTStack(files_only, outfile=vrt_file, subdataset="slc/data")

    assert s.shape == slc_stack.shape
    return s


@pytest.fixture
def vrt_stack_nc_wgs84(tmp_path, slc_stack, slc_file_list_nc_wgs84):
    # Check an alternative projection system
    vrt_file = tmp_path / "test_nc_wgs84.vrt"
    s = VRTStack(slc_file_list_nc_wgs84, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    return s


def test_create(vrt_stack, vrt_stack_nc):
    for v in [vrt_stack, vrt_stack_nc]:
        vrt_file = vrt_stack.outfile
        assert vrt_file.exists()
        assert vrt_file.stat().st_size > 0

        # Check that the VRT is valid
        ds = gdal.Open(str(vrt_file))
        assert ds is not None
        ds = None


def test_create_over_existing(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    VRTStack(slc_file_list, outfile=vrt_file)
    with pytest.raises(FileExistsError):
        VRTStack(slc_file_list, outfile=vrt_file)


def test_from_vrt_file(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)
    s2 = VRTStack.from_vrt_file(vrt_file)
    assert s == s2


def test_read_stack(vrt_stack, slc_stack):
    ds = gdal.Open(str(vrt_stack.outfile))
    loaded = ds.ReadAsArray()
    npt.assert_array_almost_equal(loaded, slc_stack)
    npt.assert_array_almost_equal(vrt_stack.read_stack(), slc_stack)


def test_read_stack_nc(vrt_stack_nc, slc_stack):
    ds = gdal.Open(str(vrt_stack_nc.outfile))
    loaded = ds.ReadAsArray()
    npt.assert_array_almost_equal(loaded, slc_stack)
    npt.assert_array_almost_equal(vrt_stack_nc.read_stack(), slc_stack)


def test_sort_order(tmp_path, slc_file_list):
    random_order = [Path(f) for f in np.random.permutation(slc_file_list)]
    # Make sure the files are sorted by date
    vrt_stack = VRTStack(random_order, outfile=tmp_path / "test.vrt")
    assert vrt_stack.file_list == [Path(f) for f in slc_file_list]

    vrt_stack2 = VRTStack(
        random_order, sort_files=False, outfile=tmp_path / "test2.vrt"
    )
    assert vrt_stack2.file_list == random_order


def test_dates(vrt_stack, vrt_stack_nc_subdataset):
    dates = vrt_stack.dates
    assert len(dates) == len(vrt_stack)
    d0 = 20220101
    for d in dates:
        assert d[0].strftime("%Y%m%d") == str(d0)
        d0 += 1

    d0 = 20220101
    for d in vrt_stack_nc_subdataset.dates:
        assert d[0].strftime("%Y%m%d") == str(d0)
        d0 += 1


def test_bad_sizes(slc_file_list, raster_10_by_20):
    from dolphin.io import get_raster_xysize

    # Make sure the files are the same size
    assert get_raster_xysize(slc_file_list[0]) == get_raster_xysize(slc_file_list[1])
    assert get_raster_xysize(slc_file_list[0]) != get_raster_xysize(raster_10_by_20)
    with pytest.raises(ValueError):
        VRTStack(slc_file_list + [raster_10_by_20], outfile="other.vrt")


# TODO: target extent
# TODO: latlon_bbox


def test_add_file(vrt_stack, slc_stack):
    # Repeat the data, but create a new file
    slc = slc_stack[0]

    # Make the file in the past
    new_path_past = (vrt_stack.outfile.parent / "20000101.slc").resolve()
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        str(new_path_past), slc.shape[1], slc.shape[0], 1, gdal.GDT_CFloat32
    )
    ds.GetRasterBand(1).WriteArray(slc)
    ds = None

    # Check that the new file is added to the VRT
    vrt_stack.add_file(new_path_past)
    assert len(vrt_stack.file_list) == slc_stack.shape[0] + 1
    assert len(vrt_stack.dates) == slc_stack.shape[0] + 1

    # Make the file in the future
    new_path_future = (vrt_stack.outfile.parent / "20250101.slc").resolve()
    ds = driver.Create(
        str(new_path_future), slc.shape[1], slc.shape[0], 1, gdal.GDT_CFloat32
    )
    ds.GetRasterBand(1).WriteArray(slc)
    ds = None

    vrt_stack.add_file(new_path_future)
    assert len(vrt_stack.file_list) == slc_stack.shape[0] + 2

    ds = gdal.Open(str(vrt_stack.outfile))
    read_stack = ds.ReadAsArray()
    assert read_stack.shape[0] == slc_stack.shape[0] + 2


def test_iter_blocks(vrt_stack):
    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(5, 5))))
    # (5, 10) total shape, breaks into 5x5 blocks
    assert len(blocks) == 2
    for b in blocks:
        assert b.shape == (len(vrt_stack), 5, 5)

    blocks, slices = zip(*list(vrt_stack.iter_blocks(block_shape=(1, 2))))
    assert len(blocks) == 25
    for b in blocks:
        assert b.shape == (len(vrt_stack), 1, 2)


def test_tiled_iter_blocks(tmp_path, tiled_file_list):
    outfile = tmp_path / "stack.vrt"
    vrt_stack = VRTStack(tiled_file_list, outfile=outfile)
    max_bytes = len(vrt_stack) * 32 * 32 * 8
    blocks, slices = zip(*list(vrt_stack.iter_blocks(max_bytes=max_bytes)))
    # (100, 200) total shape, breaks into 32x32 blocks
    assert len(blocks) == len(slices) == 28
    for i, b in enumerate(blocks, start=1):
        # Account for the smaller block sizes at the ends
        if i % 7 == 0:
            # last col
            if i > 21:  # Last row
                assert b.shape == (len(vrt_stack), 4, 8)
            else:
                assert b.shape == (len(vrt_stack), 32, 8)

    max_bytes = len(vrt_stack) * 32 * 32 * 8 * 4
    blocks, slices = zip(*list(vrt_stack.iter_blocks(max_bytes=max_bytes)))
    assert len(blocks) == len(slices) == 8
