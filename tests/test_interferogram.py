from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin import io, utils
from dolphin.interferogram import (
    Network,
    VRTInterferogram,
    _create_vrt_conj,
    estimate_correlation_from_phase,
)


def test_derived_vrt_interferogram(slc_file_list):
    """Basic test that the VRT loads the same as S1 * S2.conj()."""
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])

    assert ifg.path.name == "20220101_20220102.int.vrt"
    assert io.get_raster_xysize(ifg.path) == io.get_raster_xysize(slc_file_list[0])
    assert ifg.ref_date == datetime(2022, 1, 1)
    assert ifg.sec_date == datetime(2022, 1, 2)

    arr0 = io.load_gdal(slc_file_list[0])
    arr1 = io.load_gdal(slc_file_list[1])
    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj(), rtol=1e-6)


def test_specify_dates(slc_file_list):
    ref_slc, sec_slc = slc_file_list[0:2]
    ifg = VRTInterferogram(ref_slc=ref_slc, sec_slc=sec_slc)
    assert ifg.path.name == "20220101_20220102.int.vrt"

    # Check other dates don't fail or get overwritten
    ref_date2 = datetime(2023, 2, 2)
    sec_date2 = datetime(2023, 2, 3)
    ifg = VRTInterferogram(
        ref_slc=ref_slc, sec_slc=sec_slc, ref_date=ref_date2, sec_date=sec_date2
    )
    assert ifg.ref_date == ref_date2
    assert ifg.sec_date == sec_date2
    assert ifg.path.name == "20230202_20230203.int.vrt"

    # One at a time check
    ifg = VRTInterferogram(ref_slc=ref_slc, sec_slc=sec_slc, ref_date=ref_date2)
    assert ifg.path.name == "20230202_20220102.int.vrt"
    ifg = VRTInterferogram(ref_slc=ref_slc, sec_slc=sec_slc, sec_date=sec_date2)
    assert ifg.path.name == "20220101_20230203.int.vrt"


def test_derived_vrt_interferogram_nc(slc_file_list_nc):
    ifg = VRTInterferogram(
        ref_slc=slc_file_list_nc[0], sec_slc=slc_file_list_nc[1], subdataset="data"
    )

    assert ifg.path.name == "20220101_20220102.int.vrt"
    assert io.get_raster_xysize(ifg.path) == io.get_raster_xysize(slc_file_list_nc[0])

    arr0 = io.load_gdal(slc_file_list_nc[0])
    arr1 = io.load_gdal(slc_file_list_nc[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj(), rtol=1e-6)


def test_derived_vrt_interferogram_with_subdataset(slc_file_list_nc_with_sds):
    ifg = VRTInterferogram(
        ref_slc=slc_file_list_nc_with_sds[0], sec_slc=slc_file_list_nc_with_sds[1]
    )

    assert ifg.path.name == "20220101_20220102.int.vrt"
    assert io.get_raster_xysize(ifg.path) == io.get_raster_xysize(
        slc_file_list_nc_with_sds[0]
    )

    arr0 = io.load_gdal(slc_file_list_nc_with_sds[0])
    arr1 = io.load_gdal(slc_file_list_nc_with_sds[1])

    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj(), rtol=1e-6)

    # Now try with just the path
    path1 = utils._get_path_from_gdal_str(slc_file_list_nc_with_sds[0])
    path2 = utils._get_path_from_gdal_str(slc_file_list_nc_with_sds[1])
    # it should fail if we don't pass the subdataset
    with pytest.raises(ValueError):
        ifg2 = VRTInterferogram(ref_slc=path1, sec_slc=path2)

    ifg2 = VRTInterferogram(
        ref_slc=path1,
        sec_slc=path2,
        path=ifg.path.parent / "test2.vrt",
        subdataset="data/VV",
    )
    ifg_arr2 = ifg2.load()
    npt.assert_allclose(ifg_arr2, ifg_arr, rtol=1e-6)


def test_derived_vrt_interferogram_outdir(tmp_path, slc_file_list):
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])
    assert slc_file_list[0].parent / "20220101_20220102.int.vrt" == ifg.path

    ifg = VRTInterferogram(
        ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], outdir=tmp_path
    )
    assert tmp_path / "20220101_20220102.int.vrt" == ifg.path


def test_derived_vrt_interferogram_outfile(tmpdir, slc_file_list):
    # Change directory so we dont create a file in the source directory
    with tmpdir.as_cwd():
        ifg = VRTInterferogram(
            ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], path="test_ifg.vrt"
        )
    assert Path("test_ifg.vrt") == ifg.path


# Use use four files for the tests below
@pytest.fixture()
def four_slc_files(slc_file_list):
    # starts on 20220101
    return slc_file_list[:4]


def _get_pair_stems(slc_file_pairs):
    return [
        (a.stem.strip(".slc.tif"), b.stem.strip(".slc.tif"))  # noqa: B005
        for a, b in slc_file_pairs
    ]


class TestNetwork:
    def test_single_reference_network(self, tmp_path, four_slc_files):
        n = Network(four_slc_files, reference_idx=0, outdir=tmp_path)

        assert n.slc_file_pairs[0][0] == four_slc_files[0]
        assert n.slc_file_pairs[0][1] == four_slc_files[1]

        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220101", "20220104"),
        ]
        # check the written out files:
        assert Path(tmp_path / "20220101_20220102.int.vrt").exists()
        assert Path(tmp_path / "20220101_20220103.int.vrt").exists()
        assert Path(tmp_path / "20220101_20220104.int.vrt").exists()

        n = Network(four_slc_files, reference_idx=1, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),  # still has the same order (early, late)
            ("20220102", "20220103"),
            ("20220102", "20220104"),
        ]
        assert Path(tmp_path / "20220101_20220102.int.vrt").exists()
        assert Path(tmp_path / "20220102_20220103.int.vrt").exists()
        assert Path(tmp_path / "20220102_20220104.int.vrt").exists()

    def test_limit_by_bandwidth(self, tmp_path, four_slc_files):
        n = Network(four_slc_files, max_bandwidth=1, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220102", "20220103"),
            ("20220103", "20220104"),
        ]
        n = Network(four_slc_files, max_bandwidth=2, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220102", "20220103"),
            ("20220102", "20220104"),
            ("20220103", "20220104"),
        ]
        n = Network(four_slc_files, max_bandwidth=3, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220101", "20220104"),
            ("20220102", "20220103"),
            ("20220102", "20220104"),
            ("20220103", "20220104"),
        ]
        with pytest.raises(ValueError):
            Network(four_slc_files, max_bandwidth=0)

    def test_limit_by_temporal_baseline(self, tmp_path, four_slc_files):
        with pytest.raises(ValueError):
            Network(four_slc_files, max_temporal_baseline=0)

        n = Network(four_slc_files, max_temporal_baseline=1, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220102", "20220103"),
            ("20220103", "20220104"),
        ]
        n = Network(four_slc_files, max_temporal_baseline=2, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220102", "20220103"),
            ("20220102", "20220104"),
            ("20220103", "20220104"),
        ]
        n = Network(four_slc_files, max_temporal_baseline=500, outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220101", "20220104"),
            ("20220102", "20220103"),
            ("20220102", "20220104"),
            ("20220103", "20220104"),
        ]

    def test_annual_ifgs(self):
        dates = [
            datetime(2021, 1, 1),
            datetime(2021, 2, 1),
            datetime(2022, 1, 4),
            datetime(2022, 6, 1),
        ]
        slcs = [d.strftime("%Y%m%d") for d in dates]
        assert Network.find_annuals(slcs, dates) == [
            (slcs[0], slcs[2]),
            (slcs[1], slcs[2]),
        ]
        assert Network.find_annuals(slcs, dates, buffer_days=10) == [(slcs[0], slcs[2])]

        assert Network.find_annuals(slcs[1:], dates[1:], buffer_days=10) == []

    def test_manual_indexes(self, tmp_path, four_slc_files):
        n = Network(four_slc_files, indexes=[(0, 1)], outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            ("20220101", "20220102"),
        ]

        n = Network(four_slc_files, indexes=[(0, -1), (0, 1)], outdir=tmp_path)
        assert _get_pair_stems(n.slc_file_pairs) == [
            # It should still always come back sorted
            ("20220101", "20220102"),
            ("20220101", "20220104"),
        ]

    def test_empty_error(self, four_slc_files):
        with pytest.raises(ValueError):
            Network(four_slc_files)
            Network(four_slc_files, indexes=[])
            Network(four_slc_files, max_bandwidth=0)
            Network(four_slc_files, max_temporal_baseline=0)

    def test_combination(self, four_slc_files):
        n1 = Network(four_slc_files, reference_idx=0, write=False)
        single_ref_pairs = [
            ("20220101", "20220102"),
            ("20220101", "20220103"),
            ("20220101", "20220104"),
        ]
        assert _get_pair_stems(n1.slc_file_pairs) == single_ref_pairs

        n2 = Network(
            four_slc_files, reference_idx=0, indexes=[(1, 2), (1, 3)], write=False
        )
        new_ifgs = [
            ("20220102", "20220103"),
            ("20220102", "20220104"),
        ]
        assert _get_pair_stems(n2.slc_file_pairs) == sorted(single_ref_pairs + new_ifgs)


@pytest.fixture()
def expected_3x3_cor():
    return np.ones((3, 3))


@pytest.mark.parametrize("window_size", [3, (3, 3)])
def test_correlation_from_phase_complex_input(expected_3x3_cor, window_size):
    ifg = np.arange(1, 10).reshape(3, 3).astype(np.complex64)
    result = estimate_correlation_from_phase(ifg, window_size)
    npt.assert_allclose(result, expected_3x3_cor)


@pytest.mark.parametrize("window_size", [3, (3, 3)])
def test_correlation_from_phase_square_window_phase_input(
    expected_3x3_cor, window_size
):
    phase = np.ones((3, 3))
    result = estimate_correlation_from_phase(phase, window_size)
    npt.assert_allclose(result, expected_3x3_cor)


@pytest.mark.parametrize("window_size", [3, (3, 3)])
def test_correlation_from_phase_nan_and_zero_input(expected_3x3_cor, window_size):
    ifg = np.array([[0, np.nan, 0], [1.0 + 0.0j, 1.0, np.nan], [1.0, 1.0, 1.0]])
    window_size = 3
    result = estimate_correlation_from_phase(ifg, window_size)
    expected2 = expected_3x3_cor.copy()
    expected2[0, 0] = expected2[0, 2] = 0.0
    expected2[0, 1] = expected2[1, 2] = np.nan

    npt.assert_allclose(result, expected2, equal_nan=True)


@pytest.mark.parametrize("window_size", [3, (3, 3)])
def test_correlation_from_phase_vrtinterferogram_input(window_size, slc_file_list):
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])
    estimate_correlation_from_phase(ifg, window_size)
    # just checking it loads and runs


def test_create_vrt_conj(tmp_path, slc_file_list_nc_wgs84):
    # create a VRTInterferogram
    infile = io.format_nc_filename(slc_file_list_nc_wgs84[0], "data")
    out = tmp_path / "test.vrt"
    _create_vrt_conj(infile, output_filename=out)

    assert out.exists()
    assert io.get_raster_gt(out) == io.get_raster_gt(infile)
    assert io.get_raster_xysize(out) == io.get_raster_xysize(infile)
    assert io.get_raster_crs(out) == io.get_raster_crs(infile)
    np.testing.assert_array_equal(io.load_gdal(out).conj(), io.load_gdal(infile))


def test_network_manual_dates(four_slc_files):
    Network(
        four_slc_files,
        max_bandwidth=1,
        write=False,
        dates=["20210101", "20210107", "20210101", "20210109"],
    )


def test_network_manual_wrong_len_dates(four_slc_files):
    with pytest.raises(ValueError):
        Network(
            four_slc_files, max_bandwidth=1, write=False, dates=["20210101", "20210109"]
        )


def test_network_no_verify():
    datestrs = ["20210101", "20210107", "20210108", "20210109"]
    Network(
        datestrs,
        max_bandwidth=1,
        write=False,
        verify_slcs=False,
    )


def test_network_from_ifgs():
    """Check that the `Network` can work when passing in ifgs"""
    ifg_files = ["20210101_20210107", "20210101_20210108", "20210101_20210109"]
    n = Network(
        ifg_files,
        max_bandwidth=10,
        write=False,
        verify_slcs=False,
        dates=["2021-01-07", "2021-01-08", "2021-01-09"],
    )
    assert len(n.ifg_list) == 3
    assert n.ifg_list[0].path.name == "20210107_20210108.int.vrt"
    assert n.ifg_list[1].path.name == "20210107_20210109.int.vrt"
    assert n.ifg_list[2].path.name == "20210108_20210109.int.vrt"
