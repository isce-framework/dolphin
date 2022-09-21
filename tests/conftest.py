import numpy as np
import pytest
from osgeo import gdal


@pytest.fixture
def slc_stack():
    shape = (30, 10, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape)
    # Phase doesn't matter here
    complex_data = data * np.exp(1j * np.zeros_like(data))
    return complex_data


@pytest.fixture
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
