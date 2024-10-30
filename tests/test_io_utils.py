import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

from dolphin.io._utils import repack_raster, repack_rasters


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture(params=["float32", "complex64", "uint8"])
def test_raster(request, temp_dir):
    dtype = request.param
    raster_path = temp_dir / f"test_raster_{dtype}.tif"

    # Create a test raster
    data = np.random.rand(100, 100).astype(dtype)
    if np.dtype(dtype) == np.complex64:
        data = data + 1j * np.random.rand(100, 100)

    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 1,
        "dtype": str(dtype),
        "crs": "EPSG:4326",
        "transform": rio.transform.from_bounds(0, 0, 1, 1, 100, 100),
    }

    with rio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)

    return raster_path


def test_repack_raster(test_raster, temp_dir):
    output_dir = temp_dir / "output"
    keep_bits = 10
    with rio.open(test_raster) as src:
        dtype = src.dtypes[0]

    if np.dtype(dtype) == np.uint8:
        with pytest.raises(TypeError):
            repack_raster(
                Path(test_raster),
                output_dir=output_dir,
                keep_bits=keep_bits,
                block_shape=(32, 32),
            )
        return
    output_path = repack_raster(
        Path(test_raster),
        output_dir=output_dir,
        keep_bits=keep_bits,
        block_shape=(32, 32),
    )

    assert output_path.exists()
    assert output_path.parent == output_dir

    with rio.open(test_raster) as src, rio.open(output_path) as dst:
        assert src.profile["dtype"] == dst.profile["dtype"]
        old, new = src.read(), dst.read()
        assert old.shape == new.shape
        tol = 2**keep_bits

        # Check if data is close but not exactly the same (due to keep_bits)
        np.testing.assert_allclose(old, new, atol=tol)


def test_repack_rasters(test_raster, temp_dir):
    keep_bits = 10

    # Add another to test the threaded version
    new_raster = str(test_raster) + ".copy.tif"
    shutil.copy(test_raster, new_raster)
    raster_paths = [Path(test_raster), Path(new_raster)]

    output_dir = temp_dir / "output_multiple"
    with rio.open(raster_paths[0]) as src:
        dtype = src.dtypes[0]
    if np.dtype(dtype) == np.uint8:
        with pytest.raises(TypeError):
            repack_rasters(
                raster_paths,
                output_dir=output_dir,
                keep_bits=keep_bits,
                block_shape=(32, 32),
            )
        return
    repack_rasters(
        raster_paths,
        output_dir=output_dir,
        keep_bits=keep_bits,
        block_shape=(32, 32),
    )
