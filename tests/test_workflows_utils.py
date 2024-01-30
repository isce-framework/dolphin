from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from dolphin import stack
from dolphin.io import _readers
from dolphin.workflows.single import setup_output_folder


def test_setup_output_folder(tmpdir, tiled_file_list):
    vrt_stack = _readers.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    is_compressed = [False] * len(vrt_stack.file_list)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )
    out_file_list = setup_output_folder(
        ministack,
        output_folder=Path(tmpdir),
        like_filename=vrt_stack.outfile,
        driver="GTiff",
        dtype=np.complex64,
    )
    for out_file in out_file_list:
        assert out_file.exists()
        assert out_file.suffix == ".tif"
        assert out_file.parent == tmpdir
        ds = gdal.Open(str(out_file))
        assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
        ds = None

    is_compressed[0] = True
    m2 = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )
    out_file_list = setup_output_folder(
        m2,
        output_folder=Path(tmpdir),
        like_filename=vrt_stack.outfile,
        driver="GTiff",
        dtype="float32",
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
    vrt_stack = _readers.VRTStack(tiled_file_list, outfile=tmpdir / "stack.vrt")
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * len(vrt_stack.file_list),
    )

    out_file_list = setup_output_folder(
        ministack,
        output_folder=Path(tmpdir),
        like_filename=vrt_stack.outfile,
        driver="GTiff",
        dtype=np.complex64,
        strides=strides,
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
