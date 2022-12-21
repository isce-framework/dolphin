import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

from dolphin import io, vrt


def test_load(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)


def test_get_raster_xysize(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)
    assert (200, 100) == io.get_raster_xysize(raster_100_by_200)


def test_compute_out_size():
    strides = {"x": 3, "y": 3}
    assert (2, 2) == io.compute_out_shape((6, 6), strides)

    # 1,2 more in each direction shouldn't change it
    assert (2, 2) == io.compute_out_shape((7, 7), strides)
    assert (2, 2) == io.compute_out_shape((8, 8), strides)

    # 1,2 fewer should bump down to 1
    assert (1, 1) == io.compute_out_shape((5, 5), strides)
    assert (1, 1) == io.compute_out_shape((4, 4), strides)


def test_save_like(raster_100_by_200, tmpdir):
    arr = io.load_gdal(raster_100_by_200)

    ones = np.ones_like(arr)
    save_name = tmpdir / "ones.tif"
    io.save_arr(arr=ones, like_filename=raster_100_by_200, output_name=save_name)

    ones_loaded = io.load_gdal(save_name)
    npt.assert_array_almost_equal(ones, ones_loaded)


def test_save_empty_like(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    empty_loaded = io.load_gdal(save_name)
    zeros = np.zeros_like(empty_loaded)
    npt.assert_array_almost_equal(empty_loaded, zeros)

    # TODO: test other metadata


def test_save_strided(raster_100_by_200, tmpdir):
    save_name = tmpdir / "same_size.tif"
    strides = {"x": 1, "y": 1}
    out_shape = io.compute_out_shape((100, 200), strides)
    assert out_shape == (100, 200)
    io.save_arr(
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
    io.save_arr(
        arr=None,
        like_filename=raster_100_by_200,
        shape=out_shape,
        output_name=save_name2,
    )
    xsize, ysize = io.get_raster_xysize(save_name2)
    assert (ysize, xsize) == (25, 100)


def test_save_block(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    block_loaded = io.load_gdal(save_name)
    arr = np.zeros_like(block_loaded)
    npt.assert_array_almost_equal(block_loaded, arr)

    io.save_block(
        cur_block=np.ones((20, 30)),
        filename=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    block_loaded2 = io.load_gdal(save_name)
    arr[:20, :30] = 1
    npt.assert_array_almost_equal(block_loaded2, arr)

    io.save_block(
        cur_block=np.ones((20, 30)),
        filename=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
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
    io.save_arr(arr=cpx_arr, like_filename=raster_100_by_200, output_name=save_name)
    arr_loaded = io.load_gdal(save_name)
    assert arr_loaded.dtype == np.complex64
    npt.assert_array_almost_equal(arr_loaded, cpx_arr)


def test_save_block_cpx(raster_100_by_200, cpx_arr, tmpdir):
    save_name = tmpdir / "complex_block.tif"
    # Start with empty file
    io.save_arr(
        arr=None,
        like_filename=raster_100_by_200,
        output_name=save_name,
        dtype=np.complex64,
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded == 0).all()

    io.save_block(
        cur_block=np.ones((20, 30), dtype=np.complex64),
        filename=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded[:20, :30] == 1 + 0j).all()
    assert (arr_loaded[20:, 30:] == 0).all()

    block_cpx = cpx_arr[:10, :10].copy()
    io.save_block(
        cur_block=block_cpx,
        filename=save_name,
        rows=slice(20, 30),
        cols=slice(20, 30),
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded[20:30, 20:30] == block_cpx).all()


def test_setup_output_folder(tmpdir, tiled_file_list):
    vrt_stack = vrt.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    vrt_stack.write()
    out_file_list = io.setup_output_folder(
        vrt_stack, driver="GTiff", dtype=np.complex64
    )
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        ds = None

    out_file_list = io.setup_output_folder(
        vrt_stack,
        driver="GTiff",
        dtype="float32",
        start_idx=1,
    )
    assert len(out_file_list) == len(vrt_stack) - 1
    for out_file in out_file_list:
        assert out_file.exists()
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32


def test_setup_output_folder_strided(tmpdir, tiled_file_list):
    vrt_stack = vrt.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    vrt_stack.write()

    strides = {"x": 4, "y": 2}
    out_file_list = io.setup_output_folder(
        vrt_stack, driver="GTiff", dtype=np.complex64, strides=strides
    )
    rows, cols = vrt_stack.shape[-2:]
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir

        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        assert ds.RasterXSize == cols // strides["x"]
        assert ds.RasterYSize == rows // strides["y"]
        ds = None


def test_get_nodata_mask(tmpdir):
    # Setup stack of ones
    arr = np.ones((50, 50), dtype="float32")
    path1 = tmpdir / "20200102.tif"
    io.save_arr(arr=arr, output_name=path1)

    path2 = tmpdir / "20220103.tif"
    gdal.Translate(str(path2), str(path1))
    file_list = [path1, path2]

    vrt_stack = vrt.VRTStack(file_list, outfile=tmpdir / "stack2.vrt")
    vrt_stack.write()

    m = io.get_stack_nodata_mask(
        vrt_stack.outfile, output_file=tmpdir / "mask.tif", buffer_pixels=0
    )
    assert m.sum() == 0

    m2 = io.load_gdal(tmpdir / "mask.tif")
    assert (m2 == 0).all()

    # save some nodata
    arr[:, :10] = np.nan
    io.save_arr(arr=arr, output_name=path1)
    m = io.get_stack_nodata_mask(vrt_stack.outfile, buffer_pixels=0)
    # Should still be 0
    assert m.sum() == 0

    # Now the whole stack has nodata
    io.save_arr(arr=arr, output_name=path2)
    m = io.get_stack_nodata_mask(vrt_stack.outfile, buffer_pixels=0)
    # Should still be 0
    assert m.sum() == 10 * 50

    # but with a buffer, it should be 0
    io.save_arr(arr=arr, output_name=path2)
    m = io.get_stack_nodata_mask(vrt_stack.outfile, buffer_pixels=50)
    # Should still be 0
    assert m.sum() == 0

    # TODO: the buffer isn't making it as big as i'd expect...
    # but with a buffer, it should be 0

    io.save_arr(arr=arr, output_name=path2)
    m = io.get_stack_nodata_mask(vrt_stack.outfile, buffer_pixels=10)
    # Should still be 0
    with pytest.raises(AssertionError):
        assert m.sum() == 0


def test_get_raster_block_sizes(raster_100_by_200, tiled_raster_100_by_200):
    assert io.get_raster_block_size(tiled_raster_100_by_200) == [32, 32]
    assert io.get_raster_block_size(raster_100_by_200) == [200, 1]
    # for io.get_max_block_shape, the rasters are 8 bytes per pixel
    # if we have 1 GB, the whole raster should fit in memory
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=1e9)
    assert bs == (100, 200)

    # for untiled, the block size is one line
    bs = io.get_max_block_shape(raster_100_by_200, 1, max_bytes=0)
    # The function forces at least 16 lines to be read at a time
    assert bs == (16, 200)
    bs = io.get_max_block_shape(raster_100_by_200, 1, max_bytes=8 * 17 * 200)
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

    # 200 / 32 = 6.25, so with 7, it should add a new row
    bs = io.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=7 * bytes_per_tile
    )
    assert bs == (64, 200)


def test_iter_blocks(tiled_raster_100_by_200):
    # Try the whole raster
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=1e9)
    blocks = list(io.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    assert len(blocks) == 1
    assert blocks[0].shape == (100, 200)

    # now one block at a time
    max_bytes = 8 * 32 * 32
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    blocks = list(io.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)
    # at the ends, the blocks are smaller
    assert blocks[6].shape == (32, 8)
    assert blocks[-1].shape == (4, 8)


def test_iter_blocks_rowcols(tiled_raster_100_by_200):
    # Block size that is a multiple of the raster size
    bgen = io.iter_blocks(tiled_raster_100_by_200, (10, 20), band=1, return_slices=True)
    blocks, slices = zip(*list(bgen))

    assert blocks[0].shape == (10, 20)
    for rs, cs in slices:
        assert rs.stop - rs.start == 10
        assert cs.stop - cs.start == 20

    # Non-multiple block size
    bgen = io.iter_blocks(tiled_raster_100_by_200, (32, 32), band=1, return_slices=True)
    blocks, slices = zip(*list(bgen))
    assert blocks[0].shape == (32, 32)
    for b, (rs, cs) in zip(blocks, slices):
        assert b.shape == (rs.stop - rs.start, cs.stop - cs.start)


def test_iter_nodata(
    raster_with_nan,
    raster_with_nan_block,
    raster_with_zero_block,
    tiled_raster_100_by_200,
):
    # load one block at a time
    max_bytes = 8 * 32 * 32
    bs = io.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    blocks = list(io.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)

    # One nan should be fine, will get loaded
    blocks = list(
        io.iter_blocks(raster_with_nan, bs, band=1, skip_empty=True, nodata=np.nan)
    )
    assert len(blocks) == expected_num_blocks

    # Now check entire block for a skipped block
    blocks = list(
        io.iter_blocks(
            raster_with_nan_block, bs, band=1, skip_empty=True, nodata=np.nan
        )
    )
    assert len(blocks) == expected_num_blocks - 1

    # Now check entire block for a skipped block
    blocks = list(
        io.iter_blocks(raster_with_zero_block, bs, band=1, skip_empty=True, nodata=0)
    )
    assert len(blocks) == expected_num_blocks - 1


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
