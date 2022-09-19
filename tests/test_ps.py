import numpy as np
import pytest
from osgeo import gdal

import dolphin.ps


@pytest.fixture
def slc_stack():
    shape = (30, 10, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape)
    # Phase doesn't matter here
    complex_data = data * np.exp(1j * np.zeros_like(data))
    return complex_data


@pytest.fixture
def amp_mean_file(tmp_path):
    out = tmp_path / "amp_mean.tif"
    return out


def _write_zeros(file, shape):
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(str(file), shape[1], shape[0], 1, gdal.GDT_Float32)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.zeros(shape))
    ds.SetMetadataItem("N", "0")
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

    amp_stack = np.abs(slc_stack)

    for i in range(slc_stack.shape[0]):
        # Make a new VRT file as the stack gets bigger
        slc_vrt_file = _make_slc_vrt_file(tmp_path, slc_stack[: i + 1])
        dolphin.ps.update_amp_disp(
            amp_mean_file=amp_mean_file,
            amp_disp_file=amp_disp_file,
            slc_vrt_file=slc_vrt_file,
        )
        computed_mean = _read_file(amp_mean_file)
        computed_disp = _read_file(amp_disp_file)

        mean = amp_stack[: i + 1].mean(axis=0)
        sigma = amp_stack[: i + 1].std(axis=0)

        np.testing.assert_array_almost_equal(mean, computed_mean)
        np.testing.assert_array_almost_equal(sigma / mean, computed_disp)

    # # Run on the entire stack
    # dolphin.ps.create_amp_dispersion(
    #     slc_vrt_file=slc_vrt_file,
    #     output_file=amp_disp_file,
    #     amp_mean_file=amp_mean_file,
    #     reference_band=1,
    # )
