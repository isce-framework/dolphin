from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin import Strides, io
from dolphin.io import VRTStack
from dolphin.utils import compute_out_shape


class TestLoad:
    def test_load(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        assert arr.shape == (100, 200)

    def test_get_raster_xysize(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        assert arr.shape == (100, 200)
        assert io.get_raster_xysize(raster_100_by_200) == (200, 100)

    def test_load_1_slice(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        block = io.load_gdal(raster_100_by_200, rows=slice(0, 10))
        assert block.shape == (10, 200)
        npt.assert_allclose(block, arr[:10, :])

        block = io.load_gdal(raster_100_by_200, cols=slice(10, 20))
        assert block.shape == (100, 10)
        npt.assert_allclose(block, arr[:, 10:20])

    def test_load_slices(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        block = io.load_gdal(raster_100_by_200, rows=slice(0, 10), cols=slice(0, 10))
        assert block.shape == (10, 10)
        npt.assert_allclose(block, arr[:10, :10])

        block = io.load_gdal(raster_100_by_200, rows=slice(10, 20), cols=slice(10, 20))
        assert block.shape == (10, 10)
        npt.assert_allclose(block, arr[10:20, 10:20])

    def test_load_none_slices(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        block = io.load_gdal(raster_100_by_200, rows=slice(0, 10), cols=slice(None))
        assert block.shape == (10, 200)
        npt.assert_allclose(block, arr[:10, :])

        block = io.load_gdal(
            raster_100_by_200, rows=slice(None, None, None), cols=slice(10, 20)
        )
        assert block.shape == (100, 10)
        npt.assert_allclose(block, arr[:, 10:20])

    def test_load_slice_oob(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200)
        block = io.load_gdal(raster_100_by_200, rows=slice(0, 300), cols=slice(0, 300))
        assert block.shape == (100, 200)
        npt.assert_allclose(block, arr)

        with pytest.raises(IndexError):
            block = io.load_gdal(
                raster_100_by_200, rows=slice(300, 400), cols=slice(0, 10)
            )

    def test_load_masked(self, raster_with_nan_block):
        arr = io.load_gdal(raster_with_nan_block, masked=True)
        assert isinstance(arr, np.ma.masked_array)
        assert np.ma.is_masked(arr)
        assert arr[arr.mask].size == 32 * 32
        assert np.all(arr.mask[:32, :32])

        arr = io.load_gdal(raster_with_nan_block)
        assert not isinstance(arr, np.ma.masked_array)
        assert not np.ma.is_masked(arr)
        assert np.all(np.isnan(arr[:32, :32]))

    def test_load_masked_empty_nodata(self, raster_100_by_200):
        arr = io.load_gdal(raster_100_by_200, masked=True)
        assert isinstance(arr, np.ma.masked_array)
        assert arr.mask == np.ma.nomask

    def test_load_band(self, tmp_path, slc_stack, slc_file_list):
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


class TestWriteArr:
    def test_write_arr_like(self, raster_100_by_200, tmpdir):
        arr = io.load_gdal(raster_100_by_200)

        ones = np.ones_like(arr)
        save_name = tmpdir / "ones.tif"
        io.write_arr(arr=ones, like_filename=raster_100_by_200, output_name=save_name)

        ones_loaded = io.load_gdal(save_name)
        npt.assert_array_almost_equal(ones, ones_loaded)

    def test_write_empty_like(self, raster_100_by_200, tmpdir):
        save_name = tmpdir / "empty.tif"
        io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)

        empty_loaded = io.load_gdal(save_name)
        zeros = np.zeros_like(empty_loaded)
        npt.assert_array_almost_equal(empty_loaded, zeros)

    def test_write_metadata(self, raster_100_by_200, tmpdir):
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
            arr=None,
            like_filename=raster_100_by_200,
            output_name=save_name,
            nodata=np.nan,
        )
        assert np.isnan(io.get_raster_nodata(save_name))

    def test_set_raster_nodata(self, raster_100_by_200, tmpdir):
        save_name = tmpdir / "empty_nometa.tif"
        io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)
        io.set_raster_nodata(save_name, 123)
        assert io.get_raster_nodata(save_name) == 123

    def test_write_units(self, raster_100_by_200, tmpdir):
        save_name = tmpdir / "empty_nometa.tif"
        io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)
        assert io.get_raster_units(save_name) is None

        save_name = tmpdir / "empty_nan_meters.tif"
        io.write_arr(
            arr=None,
            like_filename=raster_100_by_200,
            output_name=save_name,
            units="meters",
        )
        assert io.get_raster_units(save_name) == "meters"

    def test_set_raster_units(self, raster_100_by_200, tmpdir):
        save_name = tmpdir / "empty_nometa.tif"
        io.write_arr(arr=None, like_filename=raster_100_by_200, output_name=save_name)
        assert io.get_raster_units(save_name) is None
        io.set_raster_units(save_name, "meters")
        assert io.get_raster_units(save_name) == "meters"

    def test_save_strided_1(self, raster_100_by_200, tmpdir):
        save_name = tmpdir / "same_size.tif"
        strides = Strides(1, 1)
        out_shape = compute_out_shape((100, 200), strides)
        assert out_shape == (100, 200)
        io.write_arr(
            arr=None,
            like_filename=raster_100_by_200,
            shape=out_shape,
            output_name=save_name,
        )

        xsize, ysize = io.get_raster_xysize(save_name)
        assert (ysize, xsize) == (100, 200)

    def test_save_strided_2(self, raster_100_by_200, tmpdir):
        save_name2 = tmpdir / "smaller_size.tif"
        strides = Strides(4, 2)
        out_shape = compute_out_shape((100, 200), strides)
        assert out_shape == (25, 100)
        io.write_arr(
            arr=None,
            like_filename=raster_100_by_200,
            shape=out_shape,
            output_name=save_name2,
        )
        xsize, ysize = io.get_raster_xysize(save_name2)
        assert (ysize, xsize) == (25, 100)

    def test_save_block(self, raster_100_by_200, tmpdir):
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

    @pytest.fixture()
    def cpx_arr(self, shape=(100, 200)):
        rng = np.random.default_rng()
        arr = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return arr.astype(np.complex64)

    def test_save_cpx(self, raster_100_by_200, cpx_arr, tmpdir):
        save_name = tmpdir / "complex.tif"
        io.write_arr(
            arr=cpx_arr, like_filename=raster_100_by_200, output_name=save_name
        )
        arr_loaded = io.load_gdal(save_name)
        assert arr_loaded.dtype == np.complex64
        npt.assert_array_almost_equal(arr_loaded, cpx_arr)

    def test_save_block_cpx(self, raster_100_by_200, cpx_arr, tmpdir):
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
