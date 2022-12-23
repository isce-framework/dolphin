from pathlib import Path

import numpy.testing as npt

from dolphin import io
from dolphin.interferogram import VRTInterferogram


def test_derived_vrt_interferogram(slc_file_list):
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])

    assert "20220101_20220102.vrt" == ifg.outfile.name
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(slc_file_list[0])

    arr0 = io.load_gdal(slc_file_list[0])
    arr1 = io.load_gdal(slc_file_list[1])
    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())


def test_derived_vrt_interferogram_nc(slc_file_list_nc):
    ifg = VRTInterferogram(ref_slc=slc_file_list_nc[0], sec_slc=slc_file_list_nc[1])

    assert "20220101_20220102.vrt" == ifg.outfile.name
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(
        slc_file_list_nc[0]
    )

    arr0 = io.load_gdal(slc_file_list_nc[0])
    arr1 = io.load_gdal(slc_file_list_nc[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())


def test_derived_vrt_interferogram_with_subdataset(slc_file_list_nc_with_sds):
    ifg = VRTInterferogram(
        ref_slc=slc_file_list_nc_with_sds[0], sec_slc=slc_file_list_nc_with_sds[1]
    )

    assert "20220101_20220102.vrt" == ifg.outfile.name
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(
        slc_file_list_nc_with_sds[0]
    )

    arr0 = io.load_gdal(slc_file_list_nc_with_sds[0])
    arr1 = io.load_gdal(slc_file_list_nc_with_sds[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())


def test_derived_vrt_interferogram_outdir(tmp_path, slc_file_list):
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])
    assert slc_file_list[0].parent / "20220101_20220102.vrt" == ifg.outfile

    ifg = VRTInterferogram(
        ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], outdir=tmp_path
    )
    assert tmp_path / "20220101_20220102.vrt" == ifg.outfile


def test_derived_vrt_interferogram_outfile(tmpdir, slc_file_list):
    # Change directory so we dont create a file in the source directory
    with tmpdir.as_cwd():
        ifg = VRTInterferogram(
            ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], outfile="test_ifg.vrt"
        )
    assert Path("test_ifg.vrt") == ifg.outfile
