import numpy as np

from dolphin import io


def test_load(raster_100_by_200):
    arr = io.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)


def test_save_like(raster_100_by_200, tmpdir):
    arr = io.load_gdal(raster_100_by_200)

    ones = np.ones_like(arr)
    save_name = tmpdir / "ones.tif"
    io.save_arr_like(arr=ones, like_filename=raster_100_by_200, output_name=save_name)

    ones_loaded = io.load_gdal(save_name)
    np.testing.assert_array_almost_equal(ones, ones_loaded)


def test_save_empty_like(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr_like(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    empty_loaded = io.load_gdal(save_name)
    zeros = np.zeros_like(empty_loaded)
    np.testing.assert_array_almost_equal(empty_loaded, zeros)

    # TODO: test other metadata


def test_save_block(raster_100_by_200, tmpdir):
    save_name = tmpdir / "empty.tif"
    io.save_arr_like(arr=None, like_filename=raster_100_by_200, output_name=save_name)

    block_loaded = io.load_gdal(save_name)
    zeros = np.zeros_like(block_loaded)
    np.testing.assert_array_almost_equal(block_loaded, zeros)

    io.save_block(
        cur_block=np.ones((20, 30)),
        output_files=save_name,
        rows=slice(0, 20),
        cols=slice(0, 30),
    )
    block_loaded2 = io.load_gdal(save_name)
    zeros[:20, :30] = 1
    np.testing.assert_array_almost_equal(block_loaded2, zeros)
