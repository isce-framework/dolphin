from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from dolphin.vrt import VRTStack

# Uses the fixtures from conftest.py


@pytest.fixture
def stack(tmp_path, slc_file_list):
    vrt_file = tmp_path / "test.vrt"
    stack = VRTStack(slc_file_list, outfile=vrt_file)
    stack.write()
    return stack


def test_create(stack):
    vrt_file = stack.outfile
    assert vrt_file.exists()
    assert vrt_file.stat().st_size > 0

    # Check that the VRT is valid
    ds = gdal.Open(str(vrt_file))
    assert ds is not None
    ds = None


def test_read(stack, slc_stack):
    ds = gdal.Open(str(stack.outfile))
    read_stack = ds.ReadAsArray()
    np.testing.assert_array_almost_equal(read_stack, slc_stack)


def test_sort_order(slc_file_list):
    random_order = [Path(f) for f in np.random.permutation(slc_file_list)]
    # Make sure the files are sorted by date
    stack = VRTStack(random_order)
    assert stack.file_list == [Path(f) for f in slc_file_list]
