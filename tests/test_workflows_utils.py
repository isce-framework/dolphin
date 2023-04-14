import random
from itertools import chain
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from dolphin import stack
from dolphin.workflows import group_by_burst
from dolphin.workflows._utils import setup_output_folder


def test_group_by_burst():
    expected = {
        "t087_185678_iw2": [
            Path("t087_185678_iw2/20180210/t087_185678_iw2_20180210_VV.h5"),
            Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318_VV.h5"),
            Path("t087_185678_iw2/20180423/t087_185678_iw2_20180423_VV.h5"),
        ],
        "t087_185678_iw3": [
            Path("t087_185678_iw3/20180210/t087_185678_iw3_20180210_VV.h5"),
            Path("t087_185678_iw3/20180318/t087_185678_iw3_20180318_VV.h5"),
            Path("t087_185678_iw3/20180517/t087_185678_iw3_20180517_VV.h5"),
        ],
        "t087_185679_iw1": [
            Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            Path("t087_185679_iw1/20180318/t087_185679_iw1_20180318_VV.h5"),
        ],
    }
    in_files = list(chain.from_iterable(expected.values()))

    assert group_by_burst(in_files) == expected

    # Any order should work
    random.shuffle(in_files)
    # but the order of the lists of each key may be different
    for burst, file_list in group_by_burst(in_files).items():
        assert sorted(file_list) == sorted(expected[burst])


def test_group_by_burst_non_opera():
    with pytest.raises(ValueError, match="Could not parse burst id"):
        group_by_burst(["20200101.slc", "20200202.slc"])
        # A combination should still error
        group_by_burst(
            [
                "20200101.slc",
                Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            ]
        )


def test_setup_output_folder(tmpdir, tiled_file_list):
    vrt_stack = stack.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    out_file_list = setup_output_folder(vrt_stack, driver="GTiff", dtype=np.complex64)
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        ds = None

    out_file_list = setup_output_folder(
        vrt_stack,
        driver="GTiff",
        dtype="float32",
        start_idx=1,
    )
    assert len(out_file_list) == len(vrt_stack) - 1
    for out_file in out_file_list:
        assert out_file.exists()
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32


@pytest.mark.parametrize(
    "strides", [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 4, "y": 2}]
)
def test_setup_output_folder_strided(tmpdir, tiled_file_list, strides):
    vrt_stack = stack.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")

    out_file_list = setup_output_folder(
        vrt_stack, driver="GTiff", dtype=np.complex64, strides=strides
    )
    rows, cols = vrt_stack.shape[-2:]
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir

        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        assert ds.RasterXSize == cols // strides["x"]
        assert ds.RasterYSize == rows // strides["y"]
        ds = None
