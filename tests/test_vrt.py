from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

from dolphin.vrt import VRTStack

# Note: uses the fixtures from conftest.py


@pytest.fixture
def vrt_stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)
    s.write()

    assert s.shape == (30, 10, 10)
    return s


@pytest.fixture
def vrt_stack_nc(tmp_path, slc_file_list_nc):
    vrt_file = tmp_path / "test_nc.vrt"
    s = VRTStack(slc_file_list_nc, outfile=vrt_file)
    s.write()

    assert s.shape == (30, 10, 10)
    return s


@pytest.fixture
def vrt_stack_nc_wgs84(tmp_path, slc_file_list_nc_wgs84):
    # Check an alternative projection system
    vrt_file = tmp_path / "test_nc_wgs84.vrt"
    s = VRTStack(slc_file_list_nc_wgs84, outfile=vrt_file)
    s.write()

    assert s.shape == (30, 10, 10)
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


def test_sort_order(slc_file_list):
    random_order = [Path(f) for f in np.random.permutation(slc_file_list)]
    # Make sure the files are sorted by date
    vrt_stack = VRTStack(random_order)
    assert vrt_stack.file_list == [Path(f) for f in slc_file_list]


def test_dates(vrt_stack):
    dates = vrt_stack.dates
    assert len(dates) == len(vrt_stack)
    d0 = 20220101
    for d in dates:
        assert d.strftime("%Y%m%d") == str(d0)
        d0 += 1


# TODO: target extent
# TODO: latlon_bbox


def test_add_file(vrt_stack, slc_stack):
    # Repeat the data, but create a new file
    slc = slc_stack[0]

    # Make the file in the past
    new_path_past = (vrt_stack.outfile.parent / "20000101.slc").absolute()
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
    new_path_future = (vrt_stack.outfile.parent / "20250101.slc").absolute()
    ds = driver.Create(
        str(new_path_future), slc.shape[1], slc.shape[0], 1, gdal.GDT_CFloat32
    )
    ds.GetRasterBand(1).WriteArray(slc)
    ds = None

    vrt_stack.add_file(new_path_future)
    assert len(vrt_stack.file_list) == slc_stack.shape[0] + 2

    vrt_stack.write()
    ds = gdal.Open(str(vrt_stack.outfile))
    read_stack = ds.ReadAsArray()
    assert read_stack.shape[0] == slc_stack.shape[0] + 2


def test_iter_blocks(vrt_stack, slc_stack):
    blocks = list(vrt_stack.iter_blocks(block_shape=(5, 5)))
    # (10, 10) total shape, breaks into 5x5 blocks
    assert len(blocks) == 4
    for b in blocks:
        assert b.shape == (len(vrt_stack), 5, 5)
        assert b.shape[0] == slc_stack.shape[0]


def test_tiled_iter_blocks(tmp_path, tiled_file_list):
    outfile = tmp_path / "stack.vrt"
    vrt_stack = VRTStack(tiled_file_list, outfile=outfile)
    vrt_stack.write()
    max_bytes = len(vrt_stack) * 32 * 32 * 8
    blocks = list(vrt_stack.iter_blocks(max_bytes=max_bytes))
    # (100, 200) total shape, breaks into 32x32 blocks
    assert len(blocks) == 28
    for i, b in enumerate(blocks, start=1):
        if i % 7 == 0:
            # last col
            if i > 21:  # Last row
                assert b.shape == (len(vrt_stack), 4, 8)
            else:
                assert b.shape == (len(vrt_stack), 32, 8)

    max_bytes = len(vrt_stack) * 32 * 32 * 8 * 4
    blocks = list(vrt_stack.iter_blocks(max_bytes=max_bytes))
    assert len(blocks) == 8
