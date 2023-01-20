from pathlib import Path

import numpy.testing as npt
import pytest

from dolphin import io
from dolphin.interferogram import Network, VRTInterferogram


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


@pytest.fixture
def slc_list():
    return ["20220101", "20220201", "20220301", "20220401"]


def test_single_reference_network(slc_list):
    n = Network(slc_list, reference_idx=0)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
    ]
    n = Network(slc_list, reference_idx=1)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
    ]


def test_limit_by_bandwidth(slc_list):
    n = Network(slc_list, max_bandwidth=1)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220301", "20220401"),
    ]
    n = Network(slc_list, max_bandwidth=2)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]
    n = Network(slc_list, max_bandwidth=3)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]


def test_limit_by_temporal_baseline(slc_list):
    n = Network(slc_list, max_temporal_baseline=1)
    assert n.ifg_list == []

    n = Network(slc_list, max_temporal_baseline=31)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220201", "20220301"),
        ("20220301", "20220401"),
    ]
    n = Network(slc_list, max_temporal_baseline=61)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]
    n = Network(slc_list, max_temporal_baseline=500)
    assert n.ifg_list == [
        ("20220101", "20220201"),
        ("20220101", "20220301"),
        ("20220101", "20220401"),
        ("20220201", "20220301"),
        ("20220201", "20220401"),
        ("20220301", "20220401"),
    ]


@pytest.fixture
def slc_list_paths(slc_list):
    return [Path(f) for f in slc_list]


def test_path_inputs(slc_list_paths):
    n = Network(slc_list_paths, max_temporal_baseline=31)
    assert n.ifg_list == [
        (Path("20220101"), Path("20220201")),
        (Path("20220201"), Path("20220301")),
        (Path("20220301"), Path("20220401")),
    ]
