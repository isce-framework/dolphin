from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import dolphin._blocks
from dolphin import io
from dolphin.stack import VRTStack


def test_load(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)


def test_get_raster_xysize(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)
    assert (200, 100) == io.get_raster_xysize(raster_100_by_200)


def test_load_1_slice(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    block = io.load_gdal(raster_100_by_200, rows=slice(0, 10))
    assert block.shape == (10, 200)
    npt.assert_allclose(block, arr[:10, :])

    block = io.load_gdal(raster_100_by_200, cols=slice(10, 20))
    assert block.shape == (100, 10)
    npt.assert_allclose(block, arr[:, 10:20])


def test_load_slices(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    block = io.load_gdal(raster_100_by_200, rows=slice(0, 10), cols=slice(0, 10))
    assert block.shape == (10, 10)
    npt.assert_allclose(block, arr[:10, :10])

    block = io.load_gdal(raster_100_by_200, rows=slice(10, 20), cols=slice(10, 20))
    assert block.shape == (10, 10)
    npt.assert_allclose(block, arr[10:20, 10:20])


def test_load_none_slices(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    block = io.load_gdal(raster_100_by_200, rows=slice(0, 10), cols=slice(None))
    assert block.shape == (10, 200)
    npt.assert_allclose(block, arr[:10, :])

    block = io.load_gdal(
        raster_100_by_200, rows=slice(None, None, None), cols=slice(10, 20)
    )
    assert block.shape == (100, 10)
    npt.assert_allclose(block, arr[:, 10:20])


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
    out_shape = dolphin._blocks.compute_out_shape((100, 200), strides)
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
    out_shape = dolphin._blocks.compute_out_shape((100, 200), strides)
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
    loader.notify_finished()

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
    loader.notify_finished()

    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)

    # One nan should be fine, will get loaded
    loader = io.EagerLoader(filename=raster_with_nan, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    loader.notify_finished()
    assert len(blocks) == expected_num_blocks

    # Now check entire block for a skipped block
    loader = io.EagerLoader(filename=raster_with_nan_block, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    loader.notify_finished()
    assert len(blocks) == expected_num_blocks - 1

    # Now check entire block for a skipped block
    loader = io.EagerLoader(filename=raster_with_zero_block, block_shape=bs)
    blocks, slices = zip(*list(loader.iter_blocks()))
    loader.notify_finished()
    assert len(blocks) == expected_num_blocks - 1


def test_iter_blocks_overlap(tiled_raster_100_by_200):
    # Block size that is a multiple of the raster size
    xhalf, yhalf = 4, 5
    check_out = np.zeros((100, 200))
    slices = list(
        io._slice_iterator((100, 200), (30, 30), overlaps=(2 * yhalf, 2 * xhalf))
    )

    for rs, cs in slices:
        trim_row = slice(rs.start + yhalf, rs.stop - yhalf)
        trim_col = slice(cs.start + xhalf, cs.stop - xhalf)
        check_out[trim_row, trim_col] += 1

    # Everywhere in the middle should have been touched onces by the iteration
    assert np.all(check_out[yhalf:-yhalf, xhalf:-xhalf] == 1)
    # the outside is still 0
    assert np.all(check_out[:yhalf] == 0)
    assert np.all(check_out[-yhalf:] == 0)
    assert np.all(check_out[:, :xhalf] == 0)
    assert np.all(check_out[:, -xhalf:] == 0)

    loader = io.EagerLoader(
        filename=tiled_raster_100_by_200,
        block_shape=(32, 32),
        overlaps=(2 * yhalf, 2 * xhalf),
    )
    assert hasattr(loader, "_finished_event")
    blocks, slices = zip(*list(loader.iter_blocks()))
    loader.notify_finished()
    check_out = np.zeros((100, 200), dtype="complex")
    xs, ys = 1, 1  # 1-by-1 strides
    for b, (rows, cols) in zip(blocks, slices):
        # Use the logic in `single.py`
        # TODO: figure out how to encapsulate so we test a function
        out_row_start = (rows.start + yhalf) // ys
        out_col_start = (cols.start + xhalf) // xs
        # Also need to trim the data blocks themselves
        trim_row_slice = slice(yhalf // ys, -yhalf // ys)
        trim_col_slice = slice(xhalf // xs, -xhalf // xs)
        b_trimmed = b[trim_row_slice, trim_col_slice]
        check_out[
            out_row_start : out_row_start + b_trimmed.shape[0],
            out_col_start : out_col_start + b_trimmed.shape[1],
        ] += b_trimmed

    expected = io.load_gdal(tiled_raster_100_by_200)
    npt.assert_allclose(
        check_out[yhalf:-yhalf, xhalf:-xhalf], expected[yhalf:-yhalf, xhalf:-xhalf]
    )


def test_iter_blocks_overlap_wide(tmp_path):
    xhalf, yhalf = 11, 5
    shape = (4563, 20622)
    block_shape = (16, shape[1])
    overlaps = (2 * yhalf, 2 * xhalf)
    check_out = np.zeros(shape)
    slices = list(io._slice_iterator(shape, block_shape=block_shape, overlaps=overlaps))

    for rs, cs in slices:
        trim_row = slice(rs.start + yhalf, rs.stop - yhalf)
        trim_col = slice(cs.start + xhalf, cs.stop - xhalf)
        check_out[trim_row, trim_col] += 1

    # Everywhere in the middle should have been touched onces by the iteration
    assert np.all(check_out[yhalf:-yhalf, xhalf:-xhalf] == 1)
    # the outside is still 0
    assert np.all(check_out[:yhalf] == 0)
    assert np.all(check_out[-yhalf:] == 0)
    assert np.all(check_out[:, :xhalf] == 0)
    assert np.all(check_out[:, -xhalf:] == 0)

    shape = (563, 2022)
    xs, ys = 6, 3
    arr = np.random.randn(*shape)
    tmp_file = tmp_path / "wide_raster.tif"
    io.write_arr(arr=arr, output_name=tmp_file)
    loader = io.EagerLoader(
        filename=tmp_file,
        block_shape=block_shape,
        overlaps=overlaps,
        queue_size=0,
    )
    # blocks, slices = zip(*list(loader.iter_blocks()))
    # loader.notify_finished()
    check_out = np.zeros(shape)
    max_row = 50
    for b, (rows, cols) in loader.iter_blocks():
        # Use the logic in `single.py`
        # TODO: figure out how to encapsulate so we test a function
        out_row_start = (rows.start + yhalf) // ys
        out_col_start = (cols.start + xhalf) // xs
        # Also need to trim the data blocks themselves
        trim_row_slice = slice(yhalf // ys, -yhalf // ys)
        trim_col_slice = slice(xhalf // xs, -xhalf // xs)
        b_trimmed = b[trim_row_slice, trim_col_slice]
        check_out[
            out_row_start : out_row_start + b_trimmed.shape[0],
            out_col_start : out_col_start + b_trimmed.shape[1],
        ] += b_trimmed
        if out_row_start > max_row:
            break

    loader.notify_finished()
    expected = io.load_gdal(tmp_file)[::ys, ::xs]
    npt.assert_allclose(
        check_out[yhalf // ys : -yhalf // ys, xhalf // xs : -xhalf // xs],
        expected[yhalf:-yhalf, xhalf:-xhalf],
    )


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
