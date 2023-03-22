from pathlib import Path
from typing import List

import numpy as np
import pytest
from make_netcdf import create_test_nc

from dolphin.workflows import config, s1_disp


@pytest.fixture()
def opera_slc_files(tmp_path, slc_stack) -> List[Path]:
    """Save the slc stack as a series of NetCDF files."""
    start_date = 20220101
    shape = (4, 256, 256)
    slc_stack = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(
        np.complex64
    )

    d = tmp_path / "s1_disp"
    d.mkdir()
    file_list = []

    *group_parts, ds_name = config.OPERA_DATASET_NAME.split("/")
    group = "/".join(group_parts)
    for burst_id in ["t087_185683_iw2", "t087_185684_iw2"]:
        for i in range(len(slc_stack)):
            fname = d / f"{burst_id}_{start_date + i}.h5"
            yoff = i * shape[0] / 2
            create_test_nc(
                fname,
                epsg=32615,
                data_ds_name=ds_name,
                # The "dummy" is so that two datasets are created in the file
                # otherwise GDAL doesn't respect the NETCDF:file:/path/to/nested/data
                subdir=[group, "dummy"],
                data=slc_stack[i],
                yoff=yoff,
            )
            file_list.append(Path(fname))

    return file_list


def test_s1_disp_run(opera_slc_files: List[Path], tmpdir):
    with tmpdir.as_cwd():
        cfg = config.Workflow(
            workflow_name=config.WorkflowName.SINGLE,
            cslc_file_list=opera_slc_files,
            interferogram_network=dict(
                network_type=config.InterferogramNetworkType.MANUAL_INDEX,
                indexes=[(0, -1)],
            ),
            phase_linking=dict(
                ministack_size=500,
            ),
        )
        s1_disp.run(cfg)
