import numpy as np
import pytest
from osgeo import gdal

from dolphin import io, vrt


def test_load(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)


def test_save_like(raster_100_by_200, tmpdir):
    arr = io.load_gdal(raster_100_by_200)

    ones = np.ones_like(arr)
    save_name = tmpdir / "ones.tif"
    io.save_arr(arr=ones, like_filename=raster_100_by_200, output_name=save_name)

    ones_loaded = io.load_gdal(save_name)
    np.testing.assert_array_almost_equal(ones, ones_loaded)


def test_save_empty_like(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    empty_loaded = io.load_gdal(save_name)
    zeros = np.zeros_like(empty_loaded)
    np.testing.assert_array_almost_equal(empty_loaded, zeros)

    # TODO: test other metadata


def test_save_block(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    block_loaded = io.load_gdal(save_name)
    arr = np.zeros_like(block_loaded)
    np.testing.assert_array_almost_equal(block_loaded, arr)

    io.save_block(
        cur_block=np.ones((20, 30)),
        output_files=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    block_loaded2 = io.load_gdal(save_name)
    arr[:20, :30] = 1
    np.testing.assert_array_almost_equal(block_loaded2, arr)

    io.save_block(
        cur_block=np.ones((20, 30)),
        output_files=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    block_loaded2 = io.load_gdal(save_name)
    arr[:20, :30] = 1
    np.testing.assert_array_almost_equal(block_loaded2, arr)


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
    np.testing.assert_array_almost_equal(arr_loaded, cpx_arr)


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
        output_files=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    arr_loaded = io.load_gdal(save_name)
    assert (arr_loaded[:20, :30] == 1 + 0j).all()
    assert (arr_loaded[20:, 30:] == 0).all()

    block_cpx = cpx_arr[:10, :10].copy()
    io.save_block(
        cur_block=block_cpx,
        output_files=save_name,
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
