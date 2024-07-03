from __future__ import annotations

import os
from pathlib import Path

import pytest

from dolphin.io import get_raster_xysize
from dolphin.workflows import config, ps


def test_ps_workflow_run(opera_slc_files_official: list[Path], tmpdir):
    # get the first half of the files, for just one burst stack
    file_list = opera_slc_files_official[: len(opera_slc_files_official) // 2]
    with tmpdir.as_cwd():
        cfg = config.PsWorkflow(
            cslc_file_list=file_list,
            input_options={"subdataset": "/data/VV"},
            worker_settings={
                "gpu_enabled": (os.environ.get("NUMBA_DISABLE_JIT") != "1")
            },
        )
        ps.run(cfg)

        work_dir = cfg.work_directory
        # Check the output files were made
        expected_outputs = [
            work_dir / "PS" / "ps_pixels.tif",
            work_dir / "PS" / "amp_mean.tif",
            work_dir / "PS" / "amp_dispersion.tif",
        ]
        for f in expected_outputs:
            assert f.exists()

        in_size = get_raster_xysize(f"NETCDF:{file_list[0]}:/data/VV")
        out_size = get_raster_xysize(expected_outputs[0])
        assert in_size == out_size


def test_ps_workflow_multi_burst(opera_slc_files_official: list[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.PsWorkflow(
            cslc_file_list=opera_slc_files_official,
            input_options={"subdataset": "/data/VV"},
            worker_settings={
                "gpu_enabled": (os.environ.get("NUMBA_DISABLE_JIT") != "1")
            },
            log_file=Path() / "dolphin_ps.log",
        )
        with pytest.raises(NotImplementedError):
            ps.run(cfg)
