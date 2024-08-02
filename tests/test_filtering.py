from pathlib import Path

import numpy as np
import pytest

from dolphin import filtering, io


def test_filter_long_wavelength():
    # Check filtering with ramp phase
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    unw_ifg = np.pi * (x + y)
    corr = np.ones(unw_ifg.shape, dtype=np.float32)
    bad_pixel_mask = corr < 0.5

    # Filtering
    filtered_ifg = filtering.filter_long_wavelength(
        unw_ifg, bad_pixel_mask=bad_pixel_mask, pixel_spacing=1000
    )
    np.testing.assert_allclose(
        filtered_ifg[10:-10, 10:-10],
        np.zeros(filtered_ifg[10:-10, 10:-10].shape),
        atol=1.0,
    )


def test_filter_long_wavelength_too_large_cutoff():
    # Check filtering with ramp phase
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    unw_ifg = np.pi * (x + y)
    bad_pixel_mask = np.zeros(unw_ifg.shape, dtype=bool)

    with pytest.raises(ValueError):
        filtering.filter_long_wavelength(
            unw_ifg,
            bad_pixel_mask=bad_pixel_mask,
            pixel_spacing=1,
            wavelength_cutoff=50_000,
        )


@pytest.fixture()
def unw_files(tmp_path):
    """Make series of files offset in lat/lon."""
    shape = (3, 9, 9)

    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    file_list = []
    for i in range(shape[0]):
        unw_arr = (i + 1) * np.pi * (x + y)
        fname = tmp_path / f"unw_{i}.tif"
        io.write_arr(arr=unw_arr, output_name=fname)
        file_list.append(Path(fname))

    return file_list


def test_filter(tmp_path, unw_files):
    output_dir = Path(tmp_path) / "filtered"
    filtering.filter_rasters(
        unw_filenames=unw_files,
        output_dir=output_dir,
        max_workers=1,
        wavelength_cutoff=50,
    )
