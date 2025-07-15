from __future__ import annotations

from pathlib import Path

import pytest

from dolphin.workflows import config, corrections, displacement

TEST_STATIC_FILES = [
    Path(__file__).parent
    / "data/opera-s1-static-packed/OPERA_L2_CSLC-S1-STATIC_T087-185683-IW2_20140403_S1A_v1.0.repacked.h5",  # noqa: E501
    Path(__file__).parent
    / "data/opera-s1-static-packed/OPERA_L2_CSLC-S1-STATIC_T087-185684-IW2_20140403_S1A_v1.0.repacked.h5",  # noqa: E501
]


@pytest.fixture()
def tec_files():
    data_dir = Path("tests/data")
    filenames = Path(data_dir / "tec_files.txt").read_text().splitlines()
    return [Path(__file__).parent / f for f in filenames]


# Fixture for files
@pytest.fixture()
def opera_static_files_official():
    data_dir = Path("tests/data")
    with open(data_dir / "opera_static_files_official.txt") as f:
        return f.readlines()


def test_corrections_run_single(
    opera_slc_files: list[Path],
    tec_files: list[Path],
    tmpdir,
):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files,
            input_options={"subdataset": "/data/VV"},
            output_options={"strides": {"x": 6, "y": 3}},
            interferogram_network={
                "indexes": [(0, -1)],
                "max_bandwidth": 2,
            },
            phase_linking={
                "ministack_size": 500,
            },
        )
        cfg_corrections = config.CorrectionOptions(
            ionosphere_files=tec_files,
            geometry_files=TEST_STATIC_FILES,
        )
        paths = displacement.run(cfg)
        assert paths.timeseries_paths is not None
        new_paths = corrections.run(
            cfg,
            correction_options=cfg_corrections,
            timeseries_paths=paths.timeseries_paths,
        )
        assert new_paths.ionospheric_corrections is not None
        assert all(p.exists() for p in new_paths.ionospheric_corrections)
