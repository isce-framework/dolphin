import zipfile
from pathlib import Path

import numpy as np
import pytest

from dolphin import Bbox, io, masking


@pytest.fixture()
def mask_files(tmp_path):
    """Make series of files offset in lat/lon."""
    shape = (9, 9)

    all_masked = np.zeros(shape, dtype="uint8")
    file_list = []
    for i in range(shape[0]):
        fname = tmp_path / f"mask_{i}.tif"
        arr = all_masked.copy()
        # make first third, then middle, then end "good" pixels
        rows = slice(3 * i, 3 * (i + 1))
        arr[rows, :] = 1
        io.write_arr(arr=arr, output_name=fname)
        file_list.append(Path(fname))

    return file_list


@pytest.mark.parametrize("convention", [None, masking.MaskConvention.ZERO_IS_NODATA])
def test_combine_mask_files(mask_files, convention):
    output_file = mask_files[0].parent / "combined.tif"
    masking.combine_mask_files(
        mask_files=mask_files,
        output_file=output_file,
        input_conventions=convention,
    )
    expected = np.zeros((9, 9), dtype="uint8")
    np.testing.assert_array_equal(expected, io.load_gdal(output_file))


def test_load_mask_as_numpy(mask_files):
    arr = masking.load_mask_as_numpy(mask_files[0])
    expected = np.ones((9, 9), dtype=bool)
    expected[:3] = False
    np.testing.assert_array_equal(arr, expected)


@pytest.fixture
def like_filename_zipped():
    return Path(__file__).parent / "data/dummy_like.tif.zip"


def test_bounds(tmp_path, like_filename_zipped):
    # Unzip to tmp_path
    with zipfile.ZipFile(like_filename_zipped, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    # Get the path of the extracted TIF file
    extracted_tif = tmp_path / "dummy_like.tif"

    output_filename = tmp_path / "mask_bounds.tif"
    bounds = Bbox(
        left=-122.90334860812246,
        bottom=51.7323987260125,
        right=-122.68416491724179,
        top=51.95333755674119,
    )
    bounds_wkt = None
    masking.create_bounds_mask(
        bounds, bounds_wkt, like_filename=extracted_tif, output_filename=output_filename
    )
    # Check result
    mask = io.load_gdal(output_filename)
    assert (mask[1405:3856, 9681:12685] == 1).all()
    # WGS84 box is not a box in UTM
    assert (mask[:1400, :] == 0).all()
    assert (mask[4000:, :] == 0).all()
    assert (mask[:, :9500] == 0).all()
    assert (mask[:, 13000:] == 0).all()
