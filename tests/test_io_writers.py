import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.errors import NotGeoreferencedWarning

from dolphin.io._core import get_raster_units, load_gdal
from dolphin.io._writers import (
    BackgroundBlockWriter,
    BackgroundRasterWriter,
    BackgroundStackWriter,
    RasterWriter,
)

# Note: uses the fixtures from conftest.py

# Get combinations of slices
slices_to_test = [slice(None), 1, slice(0, 10, 2)]


# Filter rasterio georeferencing warnings
@pytest.fixture(autouse=True)
def suppress_not_georeferenced_warning():
    """
    Pytest fixture to suppress NotGeoreferencedWarning in tests.

    This fixture automatically applies to all test functions in the module
    where it's defined, suppressing the specified warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        yield


@pytest.fixture
def output_file_list(slc_file_list):
    suffix = Path(slc_file_list[0]).suffix
    return [f"{f}_out{suffix}" for f in slc_file_list]


def test_background_block_writer(output_file_list, slc_file_list):
    from dolphin.io import write_arr

    write_arr(arr=None, output_name=output_file_list[0], like_filename=slc_file_list[0])
    data = np.random.randn(5, 10)
    w = BackgroundBlockWriter()
    rows, cols = slice(0, 5), slice(0, 10)
    w.queue_write(data, output_file_list[0], 0, 0)
    # Make sure we dont write too late
    w.notify_finished()
    assert w._thread.is_alive() is False

    assert np.allclose(load_gdal(output_file_list[0], rows=rows, cols=cols), data)


# #### RasterReader Tests ####
class TestRasterWriter:
    def raster_init(self, slc_file_list, slc_stack):
        # ignore georeferencing warnings
        with pytest.warns(rio.errors.NotGeoreferencedWarning):
            w = RasterWriter(slc_file_list[0])
        assert w.shape == slc_stack[0].shape
        assert w.dtype == slc_stack[0].dtype
        assert w.ndim == 2
        assert w.dtype == np.complex64
        assert w.filename == slc_file_list[0]
        assert w.band == 1
        assert w.closed is False
        w.close()

    def test_write(self, slc_file_list):
        data = np.random.randn(5, 10)
        w = RasterWriter(slc_file_list[0])
        rows, cols = slice(0, 5), slice(0, 10)
        w[rows, cols] = data
        w.close()

        assert np.allclose(load_gdal(slc_file_list[0], rows=rows, cols=cols), data)

    def test_context_manager(self, slc_file_list):
        rows, cols = slice(0, 5), slice(0, 10)
        data = np.random.randn(5, 10)
        with RasterWriter(slc_file_list[0]) as w:
            w[rows, cols] = data
            assert w.closed is False

        assert w.closed is True
        assert np.allclose(load_gdal(slc_file_list[0], rows=rows, cols=cols), data)


class TestBackgroundRasterWriter:
    def test_init(self, slc_file_list):
        brw = BackgroundRasterWriter(slc_file_list[0])
        with rio.open(slc_file_list[0]) as src:
            assert brw.shape == src.shape
            assert brw.dtype == src.dtypes[0]
        assert brw.ndim == 2
        assert brw.closed is False
        brw.close()
        assert brw.closed is True

    def test_setitem(self, slc_file_list):
        data = np.random.randn(5, 10)
        w = BackgroundRasterWriter(slc_file_list[0])
        rows, cols = slice(0, 5), slice(0, 10)
        w[rows, cols] = data
        # Make sure we dont write too late
        w.close()
        assert w._thread.is_alive() is False

        assert np.allclose(load_gdal(slc_file_list[0], rows=rows, cols=cols), data)

    def test_context_manager(self, slc_file_list):
        rows, cols = slice(0, 5), slice(0, 10)
        data = np.random.randn(5, 10)
        with BackgroundRasterWriter(slc_file_list[0]) as w:
            w[rows, cols] = data
            assert w.closed is False

        assert w.closed is True
        assert w._thread.is_alive() is False
        assert np.allclose(load_gdal(slc_file_list[0], rows=rows, cols=cols), data)


class TestBackgroundStackWriter:
    def test_stack(self, slc_file_list, output_file_list):
        w = BackgroundStackWriter(output_file_list, like_filename=slc_file_list[0])
        with rio.open(output_file_list[0]) as src:
            shape_3d = (len(slc_file_list), *src.shape)
            assert w.shape == shape_3d
            assert w.dtype == src.dtypes[0]

        assert w.ndim == 3
        assert w.closed is False
        w.close()
        assert w.closed is True

    def test_setitem(self, output_file_list, slc_file_list):
        data = np.random.randn(len(output_file_list), 5, 10)
        w = BackgroundStackWriter(output_file_list, like_filename=slc_file_list[0])
        rows, cols = slice(0, 5), slice(0, 10)
        w[:, rows, cols] = data
        # Make sure we dont write too late
        w.close()
        assert w._thread.is_alive() is False

        for i in range(len(output_file_list)):
            assert np.allclose(
                load_gdal(output_file_list[i], rows=rows, cols=cols), data[i]
            )

    def test_file_kwargs(self, output_file_list, slc_file_list):
        w = BackgroundStackWriter(
            output_file_list, like_filename=slc_file_list[0], units="custom units"
        )
        w.close()
        for f in output_file_list:
            assert get_raster_units(f) == "custom units"

    def test_stack_keepbits(self, slc_file_list, output_file_list):
        data = np.random.randn(len(output_file_list), 5, 10)
        rows, cols = slice(0, 5), slice(0, 10)

        w = BackgroundStackWriter(output_file_list, like_filename=slc_file_list[0])
        w[:, rows, cols] = data
        w.close()

        data2 = data.copy()
        w5 = BackgroundStackWriter(
            output_file_list, like_filename=slc_file_list[0], keep_bits=5
        )
        w5[:, rows, cols] = data2
        w5.close()
        written = load_gdal(output_file_list[0], rows=rows, cols=cols)
        # Check for the small floating point differences
        assert np.abs(written - data).max() > 1e-6
