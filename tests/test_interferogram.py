from datetime import date
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin import io, utils
from dolphin.interferogram import (
    Network,
    VRTInterferogram,
    estimate_correlation_from_phase,
)


def test_derived_vrt_interferogram(slc_file_list):
    """Basic test that the VRT loads the same as S1 * S2.conj()."""
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])

    assert "20220101_20220102.vrt" == ifg.path.name
    assert io.get_raster_xysize(ifg.path) == io.get_raster_xysize(slc_file_list[0])
    assert ifg.dates == (date(2022, 1, 1), date(2022, 1, 2))

    arr0 = io.load_gdal(slc_file_list[0])
    arr1 = io.load_gdal(slc_file_list[1])
    ifg_arr = ifg.load()
    assert ifg_arr.shape == arr0.shape
    npt.assert_allclose(ifg_arr, arr0 * arr1.conj(), rtol=1e-6)


def test_derived_vrt_interferogram_nc(slc_file_list_nc):
    ifg = VRTInterferogram(
        ref_slc=slc_file_list_nc[0], sec_slc=slc_file_list_nc[1], subdataset="data"
    )

    assert "20220101_20220102.vrt" == ifg.path.name
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

    assert "20220101_20220102.vrt" == ifg.path.name
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
        subdataset="slc/data",
    )
    ifg_arr2 = ifg2.load()
    npt.assert_allclose(ifg_arr2, ifg_arr, rtol=1e-6)


def test_derived_vrt_interferogram_outdir(tmp_path, slc_file_list):
    ifg = VRTInterferogram(ref_slc=slc_file_list[0], sec_slc=slc_file_list[1])
    assert slc_file_list[0].parent / "20220101_20220102.vrt" == ifg.path

    ifg = VRTInterferogram(
        ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], outdir=tmp_path
    )
    assert tmp_path / "20220101_20220102.vrt" == ifg.path


def test_derived_vrt_interferogram_outfile(tmpdir, slc_file_list):
    # Change directory so we dont create a file in the source directory
    with tmpdir.as_cwd():
        ifg = VRTInterferogram(
            ref_slc=slc_file_list[0], sec_slc=slc_file_list[1], path="test_ifg.vrt"
        )
    assert Path("test_ifg.vrt") == ifg.path


# Use use four files for the tests below
@pytest.fixture
def four_slc_files(slc_file_list):
    # starts on 20220101
    return slc_file_list[:4]


def _get_pair_stems(slc_file_pairs):
    return [
        (a.stem.strip(".slc.tif"), b.stem.strip(".slc.tif")) for a, b in slc_file_pairs
    ]


def test_single_reference_network(tmp_path, four_slc_files):
    n = Network(four_slc_files, reference_idx=0, outdir=tmp_path)

    assert n.slc_file_pairs[0][0] == four_slc_files[0]
    assert n.slc_file_pairs[0][1] == four_slc_files[1]

    assert _get_pair_stems(n.slc_file_pairs) == [
        ("20220101", "20220102"),
        ("20220101", "20220103"),
        ("20220101", "20220104"),
    ]
    # check the written out files:
    assert Path(tmp_path / "20220101_20220102.vrt").exists()
    assert Path(tmp_path / "20220101_20220103.vrt").exists()
    assert Path(tmp_path / "20220101_20220104.vrt").exists()

    n = Network(four_slc_files, reference_idx=1, outdir=tmp_path)
    assert _get_pair_stems(n.slc_file_pairs) == [
        ("20220101", "20220102"),  # still has the same order (early, late)
        ("20220102", "20220103"),
        ("20220102", "20220104"),
    ]
    assert Path(tmp_path / "20220101_20220102.vrt").exists()
    assert Path(tmp_path / "20220102_20220103.vrt").exists()
    assert Path(tmp_path / "20220102_20220104.vrt").exists()


def test_limit_by_bandwidth(tmp_path, four_slc_files):
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


def test_limit_by_temporal_baseline(tmp_path, four_slc_files):
    n = Network(four_slc_files, max_temporal_baseline=0, outdir=tmp_path)
    assert _get_pair_stems(n.slc_file_pairs) == []

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


def test_manual_indexes(tmp_path, four_slc_files):
    n = Network(four_slc_files, indexes=[], outdir=tmp_path)
    assert _get_pair_stems(n.slc_file_pairs) == []

    n = Network(four_slc_files, indexes=[(0, 1)], outdir=tmp_path)
    assert _get_pair_stems(n.slc_file_pairs) == [
        ("20220101", "20220102"),
    ]

    n = Network(four_slc_files, indexes=[(0, -1), (0, 1)], outdir=tmp_path)
    assert _get_pair_stems(n.slc_file_pairs) == [
        ("20220101", "20220104"),
        ("20220101", "20220102"),
    ]


@pytest.fixture
def expected_3x3_cor():
    # the edges will be less than 1 because of the windowing
    return np.array(
        [
            [0.44444444, 0.66666667, 0.44444444],
            [0.66666667, 1.0, 0.66666667],
            [0.44444444, 0.66666667, 0.44444444],
        ]
    )


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
