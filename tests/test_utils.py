from pathlib import Path

import numpy as np
import pytest

from dolphin import utils


def test_get_dates():
    assert ["20200303", "20210101"] == utils.get_dates("20200303_20210101.int")

    assert "20200303" == utils.get_dates("20200303.slc")[0]
    assert "20200303" == utils.get_dates(Path("20200303.slc"))[0]
    # Check that it's the filename, not the path
    assert "20200303" == utils.get_dates(Path("/usr/19990101/asdf20200303.tif"))[0]
    assert "20200303" == utils.get_dates("/usr/19990101/asdf20200303.tif")[0]

    assert ["20200303", "20210101"] == utils.get_dates(
        "/usr/19990101/20200303_20210101.int"
    )

    with pytest.raises(ValueError):
        utils.get_dates("/usr/19990101/notadate.tif")


def test_get_types():
    np_dtype = np.dtype("complex64")
    assert 10 == utils.numpy_to_gdal_type(np_dtype)
    assert np_dtype == utils.gdal_to_numpy_type(10)

    # round trip float32
    assert utils.gdal_to_numpy_type(utils.numpy_to_gdal_type(np.float32)) == np.float32


def test_get_raster_xysize(raster_100_by_200):
    arr = utils.load_gdal(raster_100_by_200)
    assert arr.shape == (100, 200)
    assert (200, 100) == utils.get_raster_xysize(raster_100_by_200)


def test_take_looks():
    arr = np.array([[0.1, 0.01, 2], [3, 4, 1 + 1j]])

    downsampled = utils.take_looks(arr, 2, 1, func_type="nansum")
    np.testing.assert_array_equal(downsampled, np.array([[3.1, 4.01, 3.0 + 1.0j]]))
    downsampled = utils.take_looks(arr, 2, 1, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[1.55, 2.005, 1.5 + 0.5j]]))
    downsampled = utils.take_looks(arr, 1, 2, func_type="mean")
    np.testing.assert_array_equal(downsampled, np.array([[0.055], [3.5]]))


def test_masked_looks(slc_samples):
    slc_stack = slc_samples.reshape(30, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    slc_samples_masked = slc_stack[:, ~mask]
    s1 = np.nansum(slc_samples_masked, axis=1)

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan
    s2 = np.squeeze(utils.take_looks(slc_stack_masked, 11, 11))

    np.testing.assert_array_almost_equal(s1, s2, decimal=5)


def test_get_raster_block_sizes(raster_100_by_200, tiled_raster_100_by_200):
    assert utils.get_block_size(tiled_raster_100_by_200) == [32, 32]
    assert utils.get_block_size(raster_100_by_200) == [200, 1]
    # for utils.get_max_block_shape, the rasters are 8 bytes per pixel
    # if we have 1 GB, the whole raster should fit in memory
    bs = utils.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=1e9)
    assert bs == (100, 200)

    # for untiled, the block size is one line
    bs = utils.get_max_block_shape(raster_100_by_200, 1, max_bytes=0)
    # The function forces at least 16 lines to be read at a time
    assert bs == (16, 200)
    bs = utils.get_max_block_shape(raster_100_by_200, 1, max_bytes=8 * 17 * 200)
    assert bs == (32, 200)

    # Pretend we have a stack of 10 images
    nstack = 10
    # one tile should be 8 * 32 * 32 * 10 = 81920 bytes
    bytes_per_tile = 8 * 32 * 32 * nstack
    bs = utils.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=bytes_per_tile
    )
    assert bs == (32, 32)

    # with a little more, we should get 2 tiles
    bs = utils.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=1 + bytes_per_tile
    )
    assert bs == (32, 64)

    # 200 / 32 = 6.25, so with 7, it should add a new row
    bs = utils.get_max_block_shape(
        tiled_raster_100_by_200, nstack, max_bytes=7 * bytes_per_tile
    )
    assert bs == (64, 200)


def test_iterate_blocks(tiled_raster_100_by_200):
    # Try the whole raster
    bs = utils.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=1e9)
    blocks = list(utils.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    assert len(blocks) == 1
    assert blocks[0].shape == (100, 200)

    # now one block at a time
    max_bytes = 8 * 32 * 32
    bs = utils.get_max_block_shape(tiled_raster_100_by_200, 1, max_bytes=max_bytes)
    blocks = list(utils.iter_blocks(tiled_raster_100_by_200, bs, band=1))
    row_blocks = 100 // 32 + 1
    col_blocks = 200 // 32 + 1
    expected_num_blocks = row_blocks * col_blocks
    assert len(blocks) == expected_num_blocks
    assert blocks[0].shape == (32, 32)
    # at the ends, the blocks are smaller
    assert blocks[6].shape == (32, 8)
    assert blocks[-1].shape == (4, 8)
