import numpy as np
import pytest

from dolphin import io
from dolphin.workflows._product import create_output_product

# random place in hawaii
GEOTRANSFORM = [204500.0, 5.0, 0.0, 2151300.0, 0.0, -10.0]
SRS = "EPSG:32605"
SHAPE = (256, 256)


@pytest.fixture
def unw_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "unw.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def conncomp_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.uint32)
    filename = tmp_path / "conncomp.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def tcorr_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "tcorr.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def spatial_corr_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "spatial_corr.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


def test_create_output_product(
    tmp_path,
    unw_filename,
    conncomp_filename,
    tcorr_filename,
    spatial_corr_filename,
):
    output_name = tmp_path / "output_product.nc"

    create_output_product(
        unw_filename=unw_filename,
        conncomp_filename=conncomp_filename,
        tcorr_filename=tcorr_filename,
        spatial_corr_filename=spatial_corr_filename,
        output_name=output_name,
        corrections={},
    )
