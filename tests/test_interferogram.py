import numpy.testing as npt

from dolphin import io
from dolphin.interferogram import DerivedVRTInterferogram


def test_derived_vrt_interferogram(slc_file_list):
    ifg = DerivedVRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])
    ifg.write()
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(slc_file_list[0])

    arr0 = io.load_gdal(slc_file_list[0])
    arr1 = io.load_gdal(slc_file_list[1])
    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())


def test_derived_vrt_interferogram_nc(slc_file_list_nc):
    ifg = DerivedVRTInterferogram(
        ref_slc=slc_file_list_nc[0], sec_slc=slc_file_list_nc[1]
    )
    ifg.write()
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(
        slc_file_list_nc[0]
    )

    arr0 = io.load_gdal(slc_file_list_nc[0])
    arr1 = io.load_gdal(slc_file_list_nc[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())


def test_derived_vrt_interferogram_with_subdataset(slc_file_list_nc_with_sds):
    ifg = DerivedVRTInterferogram(
        ref_slc=slc_file_list_nc_with_sds[0], sec_slc=slc_file_list_nc_with_sds[1]
    )
    ifg.write()
    assert io.get_raster_xysize(ifg.outfile) == io.get_raster_xysize(
        slc_file_list_nc_with_sds[0]
    )

    arr0 = io.load_gdal(slc_file_list_nc_with_sds[0])
    arr1 = io.load_gdal(slc_file_list_nc_with_sds[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj())
