import numpy as np
import pytest
from osgeo import gdal

from dolphin.phase_link import simulate

simulate.seed(1234)
NUM_ACQ = 30


@pytest.fixture(scope="session")
def slc_stack():
    shape = (NUM_ACQ, 10, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape)
    # Phase doesn't matter here
    complex_data = data * np.exp(1j * np.zeros_like(data))
    return complex_data


@pytest.fixture()
def slc_file_list(tmp_path, slc_stack):
    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("ENVI")
    start_date = 20220101
    name_template = tmp_path / "{date}.slc"
    file_list = []
    for i in range(shape[0]):
        fname = str(name_template).format(date=str(start_date + i))
        file_list.append(fname)
        ds = driver.Create(fname, shape[2], shape[1], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(slc_stack[i])
        ds = None
    return file_list


# Phase linking fixtures for one neighborhood tests


@pytest.fixture
def C_truth():
    C, truth = simulate.simulate_C(
        num_acq=NUM_ACQ,
        Tau0=72,
        gamma_inf=0.95,
        gamma0=0.99,
        add_signal=True,
        signal_std=0.01,
    )
    return C, truth


@pytest.fixture
def slc_samples(C_truth):
    C, _ = C_truth
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


# General utils on loading data/attributes


@pytest.fixture
def raster_100_by_200(tmp_path):
    ysize, xsize = 100, 200
    # Create a test raster
    driver = gdal.GetDriverByName("ENVI")
    filename = str(tmp_path / "test.bin")
    ds = driver.Create(filename, xsize, ysize, 1, gdal.GDT_CFloat32)
    ds.FlushCache()
    ds = None
    return filename


@pytest.fixture
def tiled_raster_100_by_200(tmp_path):
    ysize, xsize = 100, 200
    tile_size = [32, 32]
    creation_options = [
        "COMPRESS=DEFLATE",
        "ZLEVEL=5",
        "TILED=YES",
        f"BLOCKXSIZE={tile_size[0]}",
        f"BLOCKYSIZE={tile_size[1]}",
    ]
    # Create a test raster
    driver = gdal.GetDriverByName("GTiff")
    filename = str(tmp_path / "test.tif")
    ds = driver.Create(
        filename, xsize, ysize, 1, gdal.GDT_CFloat32, options=creation_options
    )
    ds.FlushCache()
    ds = None
    return filename
