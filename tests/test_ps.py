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
def slc_vrt_file(tmp_path, slc_stack):
    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("ENVI")
    data_file = tmp_path / "stack.slc"
    ds = driver.Create(str(data_file), shape[2], shape[1], shape[0], gdal.GDT_CFloat32)
    for i in range(shape[0]):
        bnd = ds.GetRasterBand(i + 1)
        bnd.WriteArray(slc_stack[i])
        bnd = None
    ds = None
    # and make a VRT for it
    vrt_file = tmp_path / "stack.slc.vrt"
    gdal.Translate(str(vrt_file), str(data_file))

    return vrt_file


@pytest.fixture
def new_slc_files(tmp_path, slc_stack):
    """Make individual SLC files for each band in the stack."""
    data_files = []
    for i in range(slc_stack.shape[0]):
        data_file = tmp_path / f"slc{i + 1}.slc"
        driver = gdal.GetDriverByName("ENVI")
        ds = driver.Create(
            str(data_file), slc_stack.shape[2], slc_stack.shape[1], 1, gdal.GDT_CFloat32
        )
        bnd = ds.GetRasterBand(1)
        bnd.WriteArray(slc_stack[i])
        bnd = None
        ds = None
        data_files.append(data_file)
    return data_files


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


def test_update_amp_disp(slc_stack, slc_vrt_file, new_slc_files):
    amp_disp_file = slc_vrt_file.parent / "amp_disp"
    amp_mean_file = slc_vrt_file.parent / "amp_mean"
    # Start out the files with all zeros, and N=0
    _write_zeros(amp_disp_file, slc_stack.shape[1:])
    _write_zeros(amp_mean_file, slc_stack.shape[1:])

    # # Run on the entire stack
    # dolphin.ps.create_amp_dispersion(
    #     slc_vrt_file=slc_vrt_file,
    #     output_file=amp_disp_file,
    #     amp_mean_file=amp_mean_file,
    #     reference_band=1,
    # )

    amp_stack = np.abs(slc_stack)

    for i in range(slc_stack.shape[0]):
        dolphin.ps.update_amp_disp(
            amp_mean_file=amp_mean_file,
            amp_disp_file=amp_disp_file,
            new_slc_file=new_slc_files[i],
        )
        computed_mean = _read_file(amp_mean_file)
        computed_disp = _read_file(amp_disp_file)

        mean = amp_stack[: i + 1].mean(axis=0)
        sigma = amp_stack[: i + 1].std(axis=0)

        np.testing.assert_array_almost_equal(mean, computed_mean)
        np.testing.assert_array_almost_equal(sigma / mean, computed_disp)
