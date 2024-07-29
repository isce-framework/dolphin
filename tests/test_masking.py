from pathlib import Path

import numpy as np
import pytest

from dolphin import io, masking


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
