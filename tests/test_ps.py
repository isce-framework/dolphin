import numpy as np
import numpy.testing as npt
import pytest
from osgeo import gdal

import dolphin.ps
from dolphin import io
from dolphin.stack import VRTStack


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


@pytest.fixture
def vrt_stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    return VRTStack(slc_file_list, outfile=vrt_file)


def test_create_ps(tmp_path, vrt_stack):
    dolphin.ps.create_ps(
        slc_vrt_file=vrt_stack.outfile,
        output_amp_dispersion_file=tmp_path / "amp_disp.tif",
        output_amp_mean_file=tmp_path / "amp_mean.tif",
        output_file=tmp_path / "ps_pixels.tif",
    )
    pass


@pytest.fixture
def vrt_stack_with_nans(tmp_path, raster_with_nan_block):
    vrt_file = tmp_path / "test_with_nans.vrt"
    return VRTStack([raster_with_nan_block, raster_with_nan_block], outfile=vrt_file)


def _write_zeros(file, shape):
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(file), shape[1], shape[0], 1, gdal.GDT_Float32)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.zeros(shape))
    ds.SetMetadataItem("N", "0", "ENVI")
    ds = bnd = None


def test_update_amp_disp(tmp_path, vrt_stack, slc_stack, slc_file_list):
    slc_vrt_file = vrt_stack.outfile
    amp_disp_file = slc_vrt_file.parent / "amp_disp"
    amp_mean_file = slc_vrt_file.parent / "amp_mean"
    # Start out the files with all zeros, and N=0
    _write_zeros(amp_disp_file, slc_stack.shape[1:])
    _write_zeros(amp_mean_file, slc_stack.shape[1:])

    out_path = tmp_path / "output"

    amp_stack = np.abs(slc_stack)

    for i in range(slc_stack.shape[0]):
        # Make a new VRT file as the stack gets bigger

        # slc_vrt_file = _make_slc_vrt_file(tmp_path, slc_stack[: i + 1])
        cur_slc_vrt_file = slc_vrt_file.parent / f"slc_{i}.vrt"
        VRTStack(slc_file_list[: i + 1], outfile=cur_slc_vrt_file)
        dolphin.ps.update_amp_disp(
            amp_mean_file=amp_mean_file,
            amp_dispersion_file=amp_disp_file,
            slc_vrt_file=cur_slc_vrt_file,
            output_directory=out_path,
        )
        new_amp_file = out_path / amp_mean_file.name
        new_disp_file = out_path / amp_disp_file.name
        computed_mean = io.load_gdal(new_amp_file)
        computed_disp = io.load_gdal(new_disp_file)

        mean = amp_stack[: i + 1].mean(axis=0)
        sigma = amp_stack[: i + 1].std(axis=0)

        npt.assert_array_almost_equal(mean, computed_mean)
        npt.assert_array_almost_equal(sigma / mean, computed_disp)

        # Move the new files to the old files
        new_amp_file.rename(amp_mean_file)
        new_disp_file.rename(amp_disp_file)
