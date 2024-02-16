from pathlib import Path

import numpy as np
import pytest

from dolphin import io
from dolphin.unwrap import _utils

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture()
def corr_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    arr = np.ones((100, 200), dtype="float32")
    # The first 20 rows have nodata
    arr[:20, :] = np.nan
    filename = d / "corr_raster.cor.tif"
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        nodata=np.nan,
        driver="GTiff",
    )
    return filename


@pytest.fixture()
def mask_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    filename = d / "mask.tif"
    arr = np.ones((100, 200), dtype=np.uint8)
    # Mask the first 20 columns
    arr[:, :20] = 0
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        driver="GTiff",
    )
    return filename


def test_create_combined_mask(corr_raster, mask_raster):
    out_raster = _utils.create_combined_mask(
        mask_filename=mask_raster, image_filename=corr_raster
    )
    assert Path(out_raster).name == "combined_mask.tif"
    mask = io.load_gdal(out_raster)
    assert (mask[:20] == 0).all()
    assert (mask[:, :20] == 0).all()
    assert (mask[20:, 20:] == 1).all()
