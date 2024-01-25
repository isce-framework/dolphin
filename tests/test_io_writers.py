import warnings

import numpy as np
import pytest
import rasterio as rio
from rasterio.errors import NotGeoreferencedWarning

from dolphin.io._core import load_gdal
from dolphin.io._writers import BackgroundRasterWriter, RasterWriter

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
        brw.notify_finished()
