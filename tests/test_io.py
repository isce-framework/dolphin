from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin import io
from dolphin.stack import VRTStack


def test_load(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)


def test_get_raster_xysize(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)
    assert (200, 100) == io.get_raster_xysize(raster_100_by_200)


def test_load_slice(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    block = io.load_gdal(raster_100_by_200, rows=slice(0, 10), cols=slice(0, 10))
    assert block.shape == (10, 10)
    npt.assert_allclose(block, arr[:10, :10])

    block = io.load_gdal(raster_100_by_200, rows=slice(10, 20), cols=slice(10, 20))
    assert block.shape == (10, 10)
    npt.assert_allclose(block, arr[10:20, 10:20])


def test_load_slice_oob(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    block = io.load_gdal(raster_100_by_200, rows=slice(0, 300), cols=slice(0, 300))
    assert block.shape == (100, 200)
    npt.assert_allclose(block, arr)

    with pytest.raises(IndexError):
        block = io.load_gdal(raster_100_by_200, rows=slice(300, 400), cols=slice(0, 10))


def test_load_masked(raster_with_nan_block):
    arr = io.load_gdal(raster_with_nan_block, masked=True)
    assert isinstance(arr, np.ma.masked_array)
    assert np.ma.is_masked(arr)
    assert arr[arr.mask].size == 32 * 32
    assert np.all(arr.mask[:32, :32])

    arr = io.load_gdal(raster_with_nan_block)
    assert not isinstance(arr, np.ma.masked_array)
    assert not np.ma.is_masked(arr)
    assert np.all(np.isnan(arr[:32, :32]))


def test_load_masked_empty_nodata(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200, masked=True)
    assert isinstance(arr, np.ma.masked_array)
    assert arr.mask == np.ma.nomask


def test_load_band(tmp_path, slc_stack, slc_file_list):
    # Check on a VRT, which has multiple bands
    vrt_file = tmp_path / "test.vrt"
    s = VRTStack(slc_file_list, outfile=vrt_file)

    assert s.shape == slc_stack.shape
    assert len(s) == len(slc_stack) == len(slc_file_list)

    arr = io.load_gdal(s.outfile)
    npt.assert_array_equal(arr, slc_stack)

    # Now load each band
    for i in range(len(slc_stack)):
        layer = io.load_gdal(s.outfile, band=i + 1)
        npt.assert_array_equal(layer, slc_stack[i])


def test_compute_out_size():
    strides = {"x": 3, "y": 3}
    assert (2, 2) == io.compute_out_shape((6, 6), strides)

    # 1,2 more in each direction shouldn't change it
    assert (2, 2) == io.compute_out_shape((7, 7), strides)
    assert (2, 2) == io.compute_out_shape((8, 8), strides)

    # 1,2 fewer should bump down to 1
    assert (1, 1) == io.compute_out_shape((5, 5), strides)
    assert (1, 1) == io.compute_out_shape((4, 4), strides)


def test_get_raster_bounds(slc_file_list_nc_wgs84):
    # Use the created WGS84 SLC
    bnds = io.get_raster_bounds(slc_file_list_nc_wgs84[0])
    expected = (-5.5, -2.0, 4.5, 3.0)
    assert bnds == expected


def test_write_arr_like(raster_100_by_200, tmpdir):
    arr = io.load_gdal(raster_100_by_200)

    ones = np.ones_like(arr)
    save_name = tmpdir / "ones.tif"
    io.write_arr(arr=ones, like_filename=raster_100_by_200, output_name=save_name)

    ones_loaded = io.load_gdal(save_name)
    npt.assert_array_almost_equal(ones, ones_loaded)


def test_write_empty_like(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    empty_loaded = io.load_gdal(save_name)
    zeros = np.zeros_like(empty_loaded)
    npt.assert_array_almost_equal(empty_loaded, zeros)


def test_write_metadata(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty_nometa.tif"
    io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)
    assert io.get_raster_dtype(save_name) == np.complex64
    assert io.get_raster_nodata(save_name) is None

    save_name = tmpdir / "empty_bool_255_nodata.tif"
    io.write_arr(
        arr=None,
        like_filename=raster_100_by_200,
        output_name=save_name,
        dtype=bool,
        nodata=255,
    )
    assert io.get_raster_nodata(save_name) == 255

    save_name = tmpdir / "empty_nan_nodata.tif"
    io.write_arr(
        arr=None, like_filename=raster_100_by_200, output_name=save_name, nodata=np.nan
    )
    assert np.isnan(io.get_raster_nodata(save_name))


def test_save_strided(raster_100_by_200, tmpdir):
    save_name = tmpdir / "same_size.tif"
    strides = {"x": 1, "y": 1}
    out_shape = io.compute_out_shape((100, 200), strides)
    assert out_shape == (100, 200)
    io.write_arr(
        arr=None,
        like_filename=raster_100_by_200,
        shape=out_shape,
        output_name=save_name,
    )

    xsize, ysize = io.get_raster_xysize(save_name)
    assert (ysize, xsize) == (100, 200)

    save_name2 = tmpdir / "smaller_size.tif"
    strides = {"x": 2, "y": 4}
    out_shape = io.compute_out_shape((100, 200), strides)
    assert out_shape == (25, 100)
    io.write_arr(
        arr=None,
        like_filename=raster_100_by_200,
        shape=out_shape,
        output_name=save_name2,
    )
    xsize, ysize = io.get_raster_xysize(save_name2)
    assert (ysize, xsize) == (25, 100)


def test_save_block(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    block_loaded = io.load_gdal(save_name)
    arr = np.zeros_like(block_loaded)
    npt.assert_array_almost_equal(block_loaded, arr)

    io.write_block(
        cur_block=np.ones((20, 30)),
        filename=save_name,
        row_start=0,
        col_start=0,
    )
    block_loaded2 = io.load_gdal(save_name)
    arr[:20, :30] = 1
    npt.assert_array_almost_equal(block_loaded2, arr)

    io.write_block(
        cur_block=np.ones((20, 30)),
        filename=save_name,
        row_start=0,
        col_start=0,
    )
    block_loaded2 = io.load_gdal(save_name)
    arr[:20, :30] = 1
    npt.assert_array_almost_equal(block_loaded2, arr)


@pytest.fixture
def cpx_arr(shape=(100, 200)):
    rng = np.random.default_rng()
    arr = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return arr.astype(np.complex64)


def test_save_cpx(raster_100_by_200, cpx_arr, tmpdir):
    save_name = tmpdir / "complex.tif"
    io.write_arr(arr=cpx_arr, like_filename=raster_100_by_200, output_name=save_name)
    arr_loaded = io.load_gdal(save_name)
    assert arr_loaded.dtype == np.complex64
    npt.assert_array_almost_equal(arr_loaded, cpx_arr)


def test_save_block_cpx(raster_100_by_200, cpx_arr, tmpdir):
    save_name = tmpdir / "complex_block.tif"
    # Start with empty file
    io.write_arr(
        arr=None,
        like_filename=raster_100_by_200,
        output_name=save_name,
        dtype=np.complex64,
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded == 0).all()

    io.write_block(
        cur_block=np.ones((20, 30), dtype=np.complex64),
        filename=save_name,
        row_start=0,
        col_start=0,
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded[:20, :30] == 1 + 0j).all()
    assert (arr_loaded[20:, 30:] == 0).all()

    block_cpx = cpx_arr[:10, :10].copy()
    io.write_block(
        cur_block=block_cpx,
        filename=save_name,
        row_start=20,
        col_start=20,
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded[20:30, 20:30] == block_cpx).all()


def test_get_raster_block_sizes(raster_100_by_200, tiled_raster_100_by_200):
    assert io.get_raster_chunk_size(tiled_raster_100_by_200) == [32, 32]
    assert io.get_raster_chunk_size(raster_100_by_200) == [200, 1]


def test_get_max_block_shape(raster_100_by_200, tiled_raster_100_by_200):
    # for io.get_max_block_shape, the rasters are 8 bytes per pixel
    # if we have 1 GB, the whole raster should fit in memory
    bs = io.get_max_block_shape(tiled_raster_100_by_200, nstack=1, max_bytes=1e9)
    assert bs == (100, 200)

    # for untiled, the block size is one line
    bs = io.get_max_block_shape(raster_100_by_200, nstack=1, max_bytes=0)
    # The function forces at least 16 lines to be read at a time
    assert bs == (16, 200)
    bs = io.get_max_block_shape(raster_100_by_200, nstack=1, max_bytes=8 * 17 * 200)
    assert bs == (32, 200)

    # Pretend we have a stack of 10 images
    nstack = 10
    # one tile should be 8 * 32 * 32 * 10 = 81920 bytes
    bytes_per_tile = 8 * 32 * 32 * nstack
    bs = io.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=bytes_per_tile
    )
    assert bs == (32, 32)

    # with a little more, we should get 2 tiles
    bs = io.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=1 + bytes_per_tile
    )
    assert bs == (32, 64)

    bs = io.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=4 * bytes_per_tile
    )
    assert bs == (64, 64)


def test_iter_blocks(tiled_raster_100_by_200):
    # Try the whole raster
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=1e9)
    loader = io.EagerLoader(filename=tiled_raster_100_by_200, block_shape=bs)
    # `list` should try to load all at once`
    block_slice_tuples = list(loader.iter_blocks())
    assert not loader._thread.is_alive()
    assert len(block_slice_tuples) == 1
    blocks, slices = zip(*list(block_slice_tuples))
    assert blocks[0].shape == (100, 200)
    rows, cols = slices[0]
    assert rows == slice(0, 100)
    assert cols == slice(0, 200)

    # now one block at a time
    max_bytes = 8 * 32 * 32
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    loader = io.EagerLoader(filename=tiled_raster_100_by_200, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))

    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)
    # at the ends, the block_slice_tuples are smaller
    assert blocks[6].shape == (32, 8)
    assert blocks[-1].shape == (4, 8)


def test_iter_blocks_rowcols(tiled_raster_100_by_200):
    # Block size that is a multiple of the raster size
    loader = io.EagerLoader(filename=tiled_raster_100_by_200, block_shape=(10, 20))
    blocks, slices = zip(*list(loader.iter_blocks()))

    assert blocks[0].shape == (10, 20)
    for rs, cs in slices:
        assert rs.stop - rs.start == 10
        assert cs.stop - cs.start == 20

    # Non-multiple block size
    loader = io.EagerLoader(filename=tiled_raster_100_by_200, block_shape=(32, 32))
    blocks, slices = zip(*list(loader.iter_blocks()))
    assert blocks[0].shape == (32, 32)
    for b, (rs, cs) in zip(blocks, slices):
        assert b.shape == (rs.stop - rs.start, cs.stop - cs.start)
    loader.notify_finished()


def test_iter_nodata(
    raster_with_nan,
    raster_with_nan_block,
    raster_with_zero_block,
    tiled_raster_100_by_200,
):
    # load one block at a time
    max_bytes = 8 * 32 * 32
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    loader = io.EagerLoader(filename=tiled_raster_100_by_200, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))

    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)
    loader.notify_finished()

    # One nan should be fine, will get loaded
    loader = io.EagerLoader(filename=raster_with_nan, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    assert len(blocks) == expected_num_blocks
    loader.notify_finished()

    # Now check entire block for a skipped block
    loader = io.EagerLoader(filename=raster_with_nan_block, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    assert len(blocks) == expected_num_blocks - 1
    loader.notify_finished()

    # Now check entire block for a skipped block
    loader = io.EagerLoader(filename=raster_with_zero_block, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    assert len(blocks) == expected_num_blocks - 1
    loader.notify_finished()


@pytest.mark.skip
def test_iter_blocks_nodata_mask(tiled_raster_100_by_200):
    # load one block at a time
    max_bytes = 8 * 32 * 32
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    blocks = list(io.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks

    nodata_mask = np.zeros((100, 200), dtype=bool)
    nodata_mask[:5, :5] = True
    # non-full-block should still all be loaded nan should be fine, will get loaded
    blocks = list(
        io.iter_blocks(
            tiled_raster_100_by_200, bs, skip_empty=True, nodata_mask=nodata_mask
        )
    )
    assert len(blocks) == expected_num_blocks

    nodata_mask[:32, :32] = True
    # non-full-block should still all be loaded nan should be fine, will get loaded
    blocks = list(
        io.iter_blocks(
            tiled_raster_100_by_200, bs, skip_empty=True, nodata_mask=nodata_mask
        )
    )
    assert len(blocks) == expected_num_blocks - 1


def test_format_nc_filename():
    expected = 'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable"'
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.nc", "variable")
        == expected
    )

    # check on Path
    assert (
        io.format_nc_filename(Path("/usr/19990101/20200303_20210101.nc"), "variable")
        == expected
    )

    # check non-netcdf file
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.tif")
        == "/usr/19990101/20200303_20210101.tif"
    )
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.int", "ignored")
        == "/usr/19990101/20200303_20210101.int"
    )

    with pytest.raises(ValueError):
        # Missing the subdataset name
        io.format_nc_filename("/usr/19990101/20200303_20210101.nc")
