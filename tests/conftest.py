from pathlib import Path

import numpy as np
import pytest
from make_netcdf import create_test_nc
from osgeo import gdal

NUM_ACQ = 30


@pytest.fixture(scope="session")
def slc_stack():
    shape = (NUM_ACQ, 5, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape) + 1j * np.random.normal(
        0, sigma, size=shape
    )
    data = data.astype(np.complex64)
    return data


@pytest.fixture()
def slc_file_list(tmp_path, slc_stack):
    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("GTiff")
    start_date = 20220101
    d = tmp_path / "gtiff"
    d.mkdir()
    name_template = d / "{date}.slc.tif"
    file_list = []
    for i in range(shape[0]):
        fname = str(name_template).format(date=str(start_date + i))
        file_list.append(Path(fname))
        ds = driver.Create(fname, shape[-1], shape[-2], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(slc_stack[i])
        ds = None

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc(tmp_path, slc_stack):
    """Save the slc stack as a series of NetCDF files."""
    start_date = 20220101
    d = tmp_path / "32615"
    d.mkdir()
    name_template = d / "{date}.nc"
    file_list = []
    for i in range(len(slc_stack)):
        fname = str(name_template).format(date=str(start_date + i))
        create_test_nc(fname, epsg=32615, subdir="/", data=slc_stack[i])
        assert 'AUTHORITY["EPSG","32615"]]' in gdal.Open(fname).GetProjection()
        file_list.append(Path(fname))

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc_wgs84(tmp_path, slc_stack):
    """Make one with lat/lon as the projection system."""

    start_date = 20220101
    d = tmp_path / "wgs84"
    d.mkdir()
    name_template = d / "{date}.nc"
    file_list = []
    for i in range(len(slc_stack)):
        fname = str(name_template).format(date=str(start_date + i))
        create_test_nc(fname, epsg=4326, subdir="/", data=slc_stack[i])
        assert 'AUTHORITY["EPSG","4326"]]' in gdal.Open(fname).GetProjection()
        file_list.append(Path(fname))

    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc_with_sds(tmp_path, slc_stack):
    """Save NetCDF files with multiple valid datsets."""
    start_date = 20220101
    d = tmp_path / "nc_with_sds"
    name_template = d / "{date}.nc"
    d.mkdir()
    file_list = []
    subdirs = ["/slc", "/slc2"]
    for i in range(len(slc_stack)):
        fname = str(name_template).format(date=str(start_date + i))
        create_test_nc(fname, epsg=32615, subdir=subdirs, data=slc_stack[i])
        # just point to one of them
        file_list.append(f"NETCDF:{fname}:/slc/data")

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list
