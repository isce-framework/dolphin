from __future__ import annotations

import os
from pathlib import Path

import pytest

from dolphin.workflows import config, displacement

# 'Grid size 49 will likely result in GPU under-utilization due to low occupancy.'
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


def test_displacement_run_single(opera_slc_files: list[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files,
            input_options=dict(subdataset="/data/VV"),
            interferogram_network=dict(
                network_type=config.InterferogramNetworkType.MANUAL_INDEX,
                indexes=[(0, -1)],
            ),
            phase_linking=dict(
                ministack_size=500,
            ),
            worker_settings=dict(
                gpu_enabled=(os.environ.get("NUMBA_DISABLE_JIT") != "1")
            ),
        )
        displacement.run(cfg)


def test_displacement_run_single_official(opera_slc_files_official: list[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files_official,
            input_options=dict(subdataset="/data/VV"),
            interferogram_network=dict(
                network_type=config.InterferogramNetworkType.MANUAL_INDEX,
                indexes=[(0, -1)],
            ),
            phase_linking=dict(
                ministack_size=500,
            ),
            worker_settings=dict(
                gpu_enabled=(os.environ.get("NUMBA_DISABLE_JIT") != "1")
            ),
        )
        displacement.run(cfg)


def test_displacement_run_stack(opera_slc_files: list[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.DisplacementWorkflow(
            cslc_file_list=opera_slc_files,
            input_options=dict(subdataset="/data/VV"),
            phase_linking=dict(
                ministack_size=500,
            ),
            worker_settings=dict(
                gpu_enabled=(os.environ.get("NUMBA_DISABLE_JIT") != "1")
            ),
            benchmark_log_dir=Path("."),
            log_file=Path(".") / "dolphin.log",
        )
        displacement.run(cfg)
