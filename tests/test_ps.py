import numpy as np
import pytest
from osgeo import gdal

import dolphin.ps
from dolphin import io


def test_ps_block(slc_stack):
    # Run the PS selector on entire stack
    amp_mean, amp_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=0.25,  # should be too low for random data
    )
    assert amp_mean.shape == amp_disp.shape == ps_pixels.shape
    assert amp_mean.dtype == amp_disp.dtype == np.float32
    assert ps_pixels.dtype == bool

    assert ps_pixels.sum() == 0

    assert amp_mean.min() > 0
    assert amp_disp.min() >= 0


def test_ps_nodata(slc_stack):
    s_nan = slc_stack.copy()
    s_nan[:, 0, 0] = np.nan
    # Run the PS selector on entire stack
    amp_mean, amp_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(s_nan),
        amp_dispersion_threshold=0.95,  # high thresh shouldn't matter for nodata
    )
    assert amp_mean[0, 0] == 0
    assert amp_disp[0, 0] == 0
    assert not ps_pixels[0, 0]


def test_ps_threshold(slc_stack):
    _, _, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=100000,
    )
    assert ps_pixels.sum() == ps_pixels.size
    _, _, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=0,
    )
    assert ps_pixels.sum() == 0


@pytest.fixture()
def vrt_stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    return io.VRTStack(slc_file_list, outfile=vrt_file)


def test_create_ps(tmp_path, vrt_stack):
    ps_mask_file = tmp_path / "ps_pixels.tif"

    amp_dispersion_file = tmp_path / "amp_disp.tif"
    amp_mean_file = tmp_path / "amp_mean.tif"
    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_amp_dispersion_file=amp_dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
    )
    assert io.get_raster_dtype(ps_mask_file) == np.uint8
    assert io.get_raster_dtype(amp_mean_file) == np.float32
    assert io.get_raster_dtype(amp_dispersion_file) == np.float32


@pytest.fixture()
def vrt_stack_with_nans(tmp_path, raster_with_nan_block):
    vrt_file = tmp_path / "test_with_nans.vrt"
    return io.VRTStack([raster_with_nan_block, raster_with_nan_block], outfile=vrt_file)


def _write_zeros(file, shape):
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(file), shape[1], shape[0], 1, gdal.GDT_Float32)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.zeros(shape))
    ds.SetMetadataItem("N", "0", "ENVI")
    ds = bnd = None


def test_multilook_ps_file(tmp_path, vrt_stack):
    ps_mask_file = tmp_path / "ps_pixels.tif"

    amp_dispersion_file = tmp_path / "amp_disp.tif"
    amp_mean_file = tmp_path / "amp_mean.tif"
    dolphin.ps.create_ps(
        reader=vrt_stack,
        like_filename=vrt_stack.outfile,
        output_amp_dispersion_file=amp_dispersion_file,
        output_amp_mean_file=amp_mean_file,
        output_file=ps_mask_file,
    )
    output_ps_file, output_amp_disp_file = dolphin.ps.multilook_ps_files(
        strides={"x": 5, "y": 3},
        ps_mask_file=ps_mask_file,
        amp_dispersion_file=amp_dispersion_file,
    )
    assert io.get_raster_dtype(output_ps_file) == np.uint8
    assert io.get_raster_dtype(output_amp_disp_file) == np.float32
