from pathlib import Path

import numpy as np
import pytest
from make_netcdf import create_test_nc
from osgeo import gdal

from dolphin.io import load_gdal, write_arr
from dolphin.phase_link import simulate

simulate._seed(1234)
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


# Phase linking fixtures for one neighborhood tests


@pytest.fixture(scope="session")
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
    d = tmp_path / "raster_100_by_200"
    d.mkdir()
    filename = str(d / "test.bin")
    ds = driver.Create(filename, xsize, ysize, 1, gdal.GDT_CFloat32)  # noqa
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
    d = tmp_path / "tiled"
    d.mkdir()
    filename = d / "20220101test.tif"
    ds = driver.Create(
        str(filename), xsize, ysize, 1, gdal.GDT_CFloat32, options=creation_options
    )
    ds.FlushCache()
    ds = None
    return filename


@pytest.fixture
def tiled_file_list(tiled_raster_100_by_200):
    tmp_path = tiled_raster_100_by_200.parent
    outname2 = tmp_path / "20220102test.tif"
    gdal.Translate(str(outname2), str(tiled_raster_100_by_200))
    outname3 = tmp_path / "20220103test.tif"
    gdal.Translate(str(outname3), str(tiled_raster_100_by_200))
    return [tiled_raster_100_by_200, outname2, outname3]


@pytest.fixture()
def raster_10_by_20(tmp_path, tiled_raster_100_by_200):
    # Write a small image to a file
    d = tmp_path / "raster_10_by_20"
    d.mkdir()
    outname2 = d / "20220102small.tif"
    gdal.Translate(str(outname2), str(tiled_raster_100_by_200), height=10, width=20)
    return outname2


@pytest.fixture
def raster_with_nan(tmpdir, tiled_raster_100_by_200):
    # Raster with one nan pixel
    start_arr = load_gdal(tiled_raster_100_by_200)
    nan_arr = start_arr.copy()
    nan_arr[0, 0] = np.nan
    output_name = tmpdir / "with_one_nan.tif"
    write_arr(
        arr=nan_arr,
        like_filename=tiled_raster_100_by_200,
        output_name=output_name,
        nodata=np.nan,
    )
    return output_name


@pytest.fixture
def raster_with_nan_block(tmpdir, tiled_raster_100_by_200):
    # One full block of 32x32 is nan
    output_name = tmpdir / "with_nans.tif"
    nan_arr = load_gdal(tiled_raster_100_by_200)
    nan_arr[:32, :32] = np.nan
    write_arr(
        arr=nan_arr,
        like_filename=tiled_raster_100_by_200,
        output_name=output_name,
        nodata=np.nan,
    )
    return output_name


@pytest.fixture
def raster_with_zero_block(tmpdir, tiled_raster_100_by_200):
    # One full block of 32x32 is nan
    output_name = tmpdir / "with_zeros.tif"
    out_arr = load_gdal(tiled_raster_100_by_200)
    out_arr[:] = 1.0

    out_arr[:32, :32] = 0
    write_arr(
        arr=out_arr,
        like_filename=tiled_raster_100_by_200,
        output_name=output_name,
        nodata=0,
    )
    return output_name
