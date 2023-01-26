import numpy as np
import numpy.testing as npt
from osgeo import gdal

import dolphin.ps


def test_ps_block(slc_stack):
    # Run the PS selector on entire stack
    amp_mean, amp_disp, ps_pixels = dolphin.ps.calc_ps_block(
        np.abs(slc_stack),
        amp_dispersion_threshold=0.35,  # should be too low for random data
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
        amp_dispersion_threshold=0.35,
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


def _write_zeros(file, shape):
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(file), shape[1], shape[0], 1, gdal.GDT_Float32)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.zeros(shape))
    ds.SetMetadataItem("N", "0", "ENVI")
    ds = bnd = None


def _read_file(file):
    ds = gdal.Open(str(file))
    data = ds.ReadAsArray()
    ds = None
    return data


def _make_slc_vrt_file(tmp_path, slc_stack):
    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("ENVI")
    data_file = tmp_path / f"stack{len(slc_stack)}.slc"
    ds = driver.Create(str(data_file), shape[2], shape[1], shape[0], gdal.GDT_CFloat32)
    for i in range(shape[0]):
        bnd = ds.GetRasterBand(i + 1)
        bnd.WriteArray(slc_stack[i])
        bnd = None
    ds = None
    # and make a VRT for it
    vrt_file = data_file.with_suffix(".slc.vrt")
    gdal.Translate(str(vrt_file), str(data_file))

    return vrt_file


def test_update_amp_disp(tmp_path, slc_stack):
    slc_vrt_file = _make_slc_vrt_file(tmp_path, slc_stack[:1])
    amp_disp_file = slc_vrt_file.parent / "amp_disp"
    amp_mean_file = slc_vrt_file.parent / "amp_mean"
    # Start out the files with all zeros, and N=0
    _write_zeros(amp_disp_file, slc_stack.shape[1:])
    _write_zeros(amp_mean_file, slc_stack.shape[1:])

    out_path = tmp_path / "output"

    amp_stack = np.abs(slc_stack)

    for i in range(slc_stack.shape[0]):
        # Make a new VRT file as the stack gets bigger
        slc_vrt_file = _make_slc_vrt_file(tmp_path, slc_stack[: i + 1])
        dolphin.ps.update_amp_disp(
            amp_mean_file=amp_mean_file,
            amp_dispersion_file=amp_disp_file,
            slc_vrt_file=slc_vrt_file,
            output_directory=out_path,
        )
        new_amp_file = out_path / amp_mean_file.name
        new_disp_file = out_path / amp_disp_file.name
        computed_mean = _read_file(new_amp_file)
        computed_disp = _read_file(new_disp_file)

        mean = amp_stack[: i + 1].mean(axis=0)
        sigma = amp_stack[: i + 1].std(axis=0)

        npt.assert_array_almost_equal(mean, computed_mean)
        npt.assert_array_almost_equal(sigma / mean, computed_disp)

        # Move the new files to the old files
        new_amp_file.rename(amp_mean_file)
        new_disp_file.rename(amp_disp_file)
