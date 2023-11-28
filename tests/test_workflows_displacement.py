from __future__ import annotations

import os
from pathlib import Path

import pytest
from opera_utils import group_by_burst

from dolphin.utils import flatten
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


def test_displacement_run_single_official_opera_naming(
    opera_slc_files_official: list[Path], tmpdir
):
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
            unwrap_options=dict(run_unwrap=False),
        )
        displacement.run(cfg)


def run_displacement_stack(path, file_list: list[Path]):
    cfg = config.DisplacementWorkflow(
        cslc_file_list=file_list,
        input_options=dict(subdataset="/data/VV"),
        work_directory=path,
        phase_linking=dict(
            ministack_size=500,
        ),
        worker_settings=dict(gpu_enabled=(os.environ.get("NUMBA_DISABLE_JIT") != "1")),
        unwrap_options=dict(run_unwrap=False),
        log_file=Path(".") / "dolphin.log",
    )
    displacement.run(cfg)


def test_stack_with_compressed(opera_slc_files, tmpdir):
    with tmpdir.as_cwd():
        p1 = Path("first_run")
        run_displacement_stack(p1, opera_slc_files)
        # Find the compressed SLC files
        new_comp_slcs = sorted(p1.rglob("compressed_*"))

        p2 = Path("second_run")
        # Add the first compressed SLC in place of first real one and run again
        by_burst = group_by_burst(opera_slc_files)
        new_real_slcs = list(flatten(v[1:] for v in by_burst.values()))
        new_file_list = new_comp_slcs + new_real_slcs

        run_displacement_stack(p2, new_file_list)

        # Now the results should be the same (for the file names)
        # check the ifg folders
        ifgs1 = sorted((p1 / "interferograms/stitched").glob("*.int"))
        ifgs2 = sorted((p2 / "interferograms/stitched").glob("*.int"))
        assert len(ifgs1) > 0
        assert [f.name for f in ifgs1] == [f.name for f in ifgs2]
