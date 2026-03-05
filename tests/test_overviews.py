import pytest
import rasterio as rio

from dolphin._overviews import (
    ImageType,
    Resampling,
    create_image_overviews,
    create_overviews,
)

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


def get_overviews(filename, band=1):
    with rio.open(filename) as src:
        return src.overviews(band)


def test_create_image_overviews(list_of_gtiff_ifgs):
    f = list_of_gtiff_ifgs[0]
    assert len(get_overviews(f)) == 0
    create_image_overviews(f, image_type=ImageType.INTERFEROGRAM)
    assert len(get_overviews(f)) > 0


def test_create_image_overviews_envi(list_of_envi_ifgs):
    f = list_of_envi_ifgs[0]
    assert len(get_overviews(f)) == 0
    create_image_overviews(f, image_type=ImageType.INTERFEROGRAM)
    assert len(get_overviews(f)) > 0


@pytest.mark.parametrize("resampling", list(Resampling))
def test_resamplings(list_of_gtiff_ifgs, resampling):
    f = list_of_gtiff_ifgs[0]
    assert len(get_overviews(f)) == 0
    create_image_overviews(f, resampling=resampling)
    assert len(get_overviews(f)) > 0


@pytest.mark.parametrize("levels", [[2, 4], [4, 8, 18]])
def test_levels(list_of_gtiff_ifgs, levels):
    f = list_of_gtiff_ifgs[0]
    assert len(get_overviews(f)) == 0
    create_image_overviews(f, resampling="nearest", levels=levels)
    assert len(get_overviews(f)) == len(levels)


def test_create_overviews(list_of_gtiff_ifgs):
    create_overviews(list_of_gtiff_ifgs, image_type=ImageType.INTERFEROGRAM)
    assert all(len(get_overviews(f)) > 0 for f in list_of_gtiff_ifgs)
