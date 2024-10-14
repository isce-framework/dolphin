from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from dolphin import stack
from dolphin.io import _readers
from dolphin.workflows._utils import parse_ionosphere_files
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


def test_parse_ionosphere_files():
    ionosphere_files = [
        Path("JPL0OPSFIN_20232780000_01D_02H_GIM.INX"),
        Path("jplg1100.23i"),
        Path("jplg1820.23i"),
        Path("JPL0OPSFIN_20232660000_01D_02H_GIM.INX"),
        Path("JPL0OPSFIN_20232300000_01D_02H_GIM.INX"),
        Path("jplg2970.16i"),
        Path("JPL0OPSFIN_20232420000_01D_02H_GIM.INX"),
        Path("JPL0OPSFIN_20232540000_01D_02H_GIM.INX"),
    ]

    expected_output = {
        (datetime(2023, 4, 20),): [Path("jplg1100.23i")],
        (datetime(2023, 7, 1),): [Path("jplg1820.23i")],
        (datetime(2016, 10, 23),): [Path("jplg2970.16i")],
        (datetime(2023, 10, 5, 0, 0, 0),): [
            Path("JPL0OPSFIN_20232780000_01D_02H_GIM.INX")
        ],
        (datetime(2023, 9, 23, 0, 0, 0),): [
            Path("JPL0OPSFIN_20232660000_01D_02H_GIM.INX")
        ],
        (datetime(2023, 8, 18, 0, 0, 0),): [
            Path("JPL0OPSFIN_20232300000_01D_02H_GIM.INX")
        ],
        (datetime(2023, 8, 30, 0, 0, 0),): [
            Path("JPL0OPSFIN_20232420000_01D_02H_GIM.INX")
        ],
        (datetime(2023, 9, 11, 0, 0, 0),): [
            Path("JPL0OPSFIN_20232540000_01D_02H_GIM.INX")
        ],
    }

    grouped_iono_files = parse_ionosphere_files(ionosphere_files)
    assert len(grouped_iono_files) == len(
        expected_output
    ), "Number of grouped dates does not match expected output."

    for date_tuple, files in expected_output.items():
        assert (
            date_tuple in grouped_iono_files
        ), f"Date {date_tuple} not found in grouped ionosphere files."
        assert (
            grouped_iono_files[date_tuple] == files
        ), f"Files for date {date_tuple} do not match expected files."
