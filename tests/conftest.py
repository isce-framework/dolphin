from __future__ import annotations

import datetime
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from make_netcdf import create_test_nc
from osgeo import gdal
from rasterio.errors import NotGeoreferencedWarning

# https://numba.readthedocs.io/en/stable/user/threading-layer.html#example-of-limiting-the-number-of-threads
if not os.environ.get("NUMBA_NUM_THREADS"):
    os.environ["NUMBA_NUM_THREADS"] = str(min(os.cpu_count(), 16))  # type: ignore[type-var]

from opera_utils import OPERA_DATASET_NAME

from dolphin.io import load_gdal, write_arr
from dolphin.phase_link import simulate

NUM_ACQ = 30


# Filter rasterio georeferencing warnings
@pytest.fixture(autouse=True)
def suppress_not_georeferenced_warning():
    """
    Pytest fixture to suppress NotGeoreferencedWarning in tests.

    This fixture automatically applies to all test functions in the module
    where it's defined, suppressing the specified warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        yield


# https://github.com/pytest-dev/pytest/issues/667#issuecomment-112206152
@pytest.fixture()
def random():
    np.random.seed(1234)
    simulate._seed(1234)


@pytest.fixture(scope="session")
def slc_stack():
    shape = (NUM_ACQ, 5, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape) + 1j * np.random.normal(
        0, sigma, size=shape
    )
    return data.astype(np.complex64)


@pytest.fixture()
def slc_date_list(slc_stack):
    start_date = datetime.datetime(2022, 1, 1)
    out = []
    dt = datetime.timedelta(days=1)
    for i in range(len(slc_stack)):
        out.append(start_date + i * dt)

    return out


@pytest.fixture()
def slc_file_list(tmp_path, slc_stack, slc_date_list):
    shape = slc_stack.shape
    # Write to a file
    driver = gdal.GetDriverByName("GTiff")
    d = tmp_path / "gtiff"
    d.mkdir()
    name_template = d / "{date}.slc.tif"

    file_list = []
    for cur_date, cur_slc in zip(slc_date_list, slc_stack):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        file_list.append(Path(fname))
        ds = driver.Create(fname, shape[-1], shape[-2], 1, gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(cur_slc)
        ds = None

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc(tmp_path, slc_stack, slc_date_list):
    """Save the slc stack as a series of NetCDF files."""
    d = tmp_path / "32615"
    d.mkdir()
    name_template = d / "{date}.nc"
    file_list = []
    for cur_date, cur_slc in zip(slc_date_list, slc_stack):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        create_test_nc(fname, epsg=32615, subdir="/", data=cur_slc)
        assert 'AUTHORITY["EPSG","32615"]]' in gdal.Open(fname).GetProjection()
        file_list.append(Path(fname))

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc_wgs84(tmp_path, slc_stack, slc_date_list):
    """Make one with lat/lon as the projection system."""

    d = tmp_path / "wgs84"
    d.mkdir()
    name_template = d / "{date}.nc"
    file_list = []
    for cur_date, cur_slc in zip(slc_date_list, slc_stack):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        create_test_nc(fname, epsg=4326, subdir="/", data=cur_slc)
        assert 'AUTHORITY["EPSG","4326"]]' in gdal.Open(fname).GetProjection()
        file_list.append(Path(fname))

    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


@pytest.fixture()
def slc_file_list_nc_with_sds(tmp_path, slc_stack, slc_date_list):
    """Save NetCDF files with multiple valid datsets."""
    d = tmp_path / "nc_with_sds"
    name_template = d / "{date}.nc"
    d.mkdir()
    file_list = []
    subdirs = ["/data", "/data2"]
    ds_name = "VV"
    for cur_date, cur_slc in zip(slc_date_list, slc_stack):
        fname = str(name_template).format(date=cur_date.strftime("%Y%m%d"))
        create_test_nc(
            fname, epsg=32615, subdir=subdirs, data_ds_name=ds_name, data=cur_slc
        )
        # just point to one of them
        file_list.append(f"NETCDF:{fname}:{subdirs[0]}/{ds_name}")

    # Write the list of SLC files to a text file
    with open(d / "slclist.txt", "w") as f:
        f.write("\n".join([str(f) for f in file_list]))
    return file_list


# Phase linking fixtures for one neighborhood tests


@pytest.fixture(scope="session")
def C_truth():
    C, truth = simulate.simulate_coh(
        num_acq=NUM_ACQ,
        Tau0=72,
        gamma_inf=0.95,
        gamma0=0.99,
        add_signal=True,
        signal_std=0.01,
    )
    return C, truth


@pytest.fixture()
def slc_samples(C_truth):
    C, _ = C_truth
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


# General utils on loading data/attributes


@pytest.fixture()
def raster_100_by_200(tmp_path):
    ysize, xsize = 100, 200
    # Create a test raster
    driver = gdal.GetDriverByName("ENVI")
    d = tmp_path / "raster_100_by_200"
    d.mkdir()
    filename = str(d / "test.bin")
    ds = driver.Create(filename, xsize, ysize, 1, gdal.GDT_CFloat32)
    data = np.random.randn(ysize, xsize) + 1j * np.random.randn(ysize, xsize)
    ds.WriteArray(data)
    ds.FlushCache()
    ds = None
    return filename


@pytest.fixture()
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
    data = np.random.randn(ysize, xsize) + 1j * np.random.randn(ysize, xsize)
    ds.WriteArray(data)
    ds.FlushCache()
    ds = None
    return filename


@pytest.fixture()
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


@pytest.fixture()
def raster_with_nan(tmp_path, tiled_raster_100_by_200):
    # Raster with one nan pixel
    start_arr = load_gdal(tiled_raster_100_by_200)
    nan_arr = start_arr.copy()
    nan_arr[0, 0] = np.nan
    output_name = tmp_path / "with_one_nan.tif"
    write_arr(
        arr=nan_arr,
        like_filename=tiled_raster_100_by_200,
        output_name=output_name,
        nodata=np.nan,
    )
    return output_name


@pytest.fixture()
def raster_with_nan_block(tmp_path, tiled_raster_100_by_200):
    # One full block of 32x32 is nan
    output_name = tmp_path / "with_nans.tif"
    nan_arr = load_gdal(tiled_raster_100_by_200)
    nan_arr[:32, :32] = np.nan
    write_arr(
        arr=nan_arr,
        like_filename=tiled_raster_100_by_200,
        output_name=output_name,
        nodata=np.nan,
    )
    return output_name


@pytest.fixture()
def raster_with_zero_block(tmp_path, tiled_raster_100_by_200):
    # One full block of 32x32 is nan
    output_name = tmp_path / "with_zeros.tif"
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


# For displacement/workflow tests


DX, DY = 5, -10
X0, Y0 = 246362.5, 2153130.0
GRID_MAPPING_DSET = "spatial_ref"


@pytest.fixture()
def opera_slc_files(tmp_path) -> list[Path]:
    """Save the slc stack as a series of NetCDF files."""
    import h5py

    start_date = 20220101
    shape = (4, 128, 128)
    slc_stack = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(
        np.complex64
    )

    d = tmp_path / "input_slcs"
    d.mkdir()
    file_list = []

    *group_parts, ds_name = OPERA_DATASET_NAME.split("/")
    group = "/".join(group_parts)
    for burst_id in ["t087_185683_iw2", "t087_185684_iw2"]:
        for i in range(len(slc_stack)):
            fname = d / f"{burst_id}_{start_date + i}_20231201.h5"
            if i == 0:
                datestr = f"{start_date}_{start_date + i}_{start_date + len(slc_stack)}"
                fname = d / f"COMPRESSED_{burst_id}_{datestr}.h5"
            yoff = Y0 + i * shape[0] / 2
            create_test_nc(
                fname,
                epsg=32605,
                data_ds_name=ds_name,
                # The "dummy" is so that two datasets are created in the file
                # otherwise GDAL doesn't respect the NETCDF:file:/path/to/nested/data
                subdir=[group, "dummy"],
                data=slc_stack[i],
                xoff=X0,
                yoff=yoff,
                dx=DX,
                dy=abs(DY),
            )
            with h5py.File(fname, "a") as hf:
                hf[
                    "/metadata/processing_information/input_burst_metadata/wavelength"
                ] = 0.0554658
                start_dt = datetime.datetime.strptime(f"{start_date + i}", "%Y%m%d")
                # 2021-07-24 00:51:33.299551
                out_fmt = "%Y-%m-%d %H:%M:%S.%f"
                hf["/identification/zero_doppler_start_time"] = start_dt.strftime(
                    out_fmt
                )
            file_list.append(Path(fname))

    return file_list


@pytest.fixture()
def opera_slc_files_official(tmp_path) -> list[Path]:
    import h5py

    base = "OPERA_L2_CSLC-S1"
    ending = "20230101T100506Z_S1A_VV_v1.0"
    # expected = {
    # "t087_185678_iw2": [
    # Path(f"{base}_T087-185678-IW2_20180210T232711Z_{ending}"),
    start = datetime.datetime(2022, 1, 1, 1, 2, 3)
    dt = datetime.timedelta(days=1)
    shape = (4, 128, 128)
    phase_stack = np.empty((4, 128, 128), dtype="float32")
    ramp0 = np.ones((128, 1)) * np.arange(128).reshape(1, -1)
    ramp0 /= 128
    for idx in range(len(phase_stack)):
        cur_slope = 0.5 * 12 * idx
        phase_stack[idx] = cur_slope * ramp0
    # Add a little noise
    noise = 0.05 * (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(
        np.complex64
    )
    slc_stack = np.exp(1j * phase_stack) + noise

    d = tmp_path / "input_slcs"
    d.mkdir()
    file_list = []

    *group_parts, ds_name = OPERA_DATASET_NAME.split("/")
    group = "/".join(group_parts)
    for burst_id in ["T087-185683-IW2", "T087-185684-IW2"]:
        for i in range(len(slc_stack)):
            date_str = (start + i * dt).strftime("%Y%m%dT%H%M%S")
            fname = d / f"{base}_{burst_id}_{date_str}_{ending}.h5"
            yoff = Y0 + i * shape[0] / 2
            create_test_nc(
                fname,
                epsg=32605,
                data_ds_name=ds_name,
                # The "dummy" is so that two datasets are created in the file
                # otherwise GDAL doesn't respect the NETCDF:file:/path/to/nested/data
                subdir=[group, "corrections"],
                data=slc_stack[i],
                yoff=yoff,
                xoff=X0,
                dx=DX,
                dy=DY,
            )
            file_list.append(Path(fname))
            with h5py.File(fname, "a") as hf:
                hf[
                    "/metadata/processing_information/input_burst_metadata/wavelength"
                ] = 0.0554658
                # "2021-07-24 00:51:33.299551"
                out_fmt = "%Y-%m-%d %H:%M:%S.%f"
                hf["/identification/zero_doppler_start_time"] = (
                    start + i * dt
                ).strftime(out_fmt)

    return file_list


# For unwrapping/overviews
@pytest.fixture()
def list_of_gtiff_ifgs(tmp_path, raster_100_by_200):
    x, y = np.meshgrid(np.arange(200), np.arange(100))
    # simulate a simple phase ramp
    phase = 0.003 * x + 0.002 * y
    ifg_list = []
    # simulate three interferograms
    constants = [1, 2, 3]
    for i in range(3):
        # Create a copy of the raster in the same directory
        f = tmp_path / f"ifg_{i}.int.tif"
        array = np.exp(1j * (phase + constants[i]))
        write_arr(
            arr=array,
            output_name=f,
            like_filename=raster_100_by_200,
            driver="GTiff",
        )
        ifg_list.append(f)

    return ifg_list


@pytest.fixture()
def list_of_envi_ifgs(tmp_path, raster_100_by_200):
    ifg_list = []
    for i in range(3):
        # Create a copy of the raster in the same directory
        f = tmp_path / f"ifg_{i}.int"
        ifg_list.append(f)
        write_arr(
            arr=np.ones((100, 200), dtype=np.complex64),
            output_name=f,
            like_filename=raster_100_by_200,
            driver="ENVI",
        )
        ifg_list.append(f)

    return ifg_list
