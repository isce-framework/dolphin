from pathlib import Path

import numpy as np
import pytest

from dolphin import filtering, io
from dolphin.goldstein import goldstein


def test_filter_long_wavelength():
    # Check filtering with ramp phase
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    unw_ifg = np.pi * (x + y)

    corr = np.ones(unw_ifg.shape, dtype=np.float32)
    bad_pixel_mask = corr < 0.5

    # Filtering
    filtered_ifg = filtering.filter_long_wavelength(
        unw_ifg,
        bad_pixel_mask=bad_pixel_mask,
        pixel_spacing=100,
        wavelength_cutoff=1_000,
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


class TestGoldstein:
    def test_goldstein_non_multiple_dimensions(self):
        """Test that edges are not zeroed when dimensions aren't multiples of psize."""
        np.random.seed(42)
        rows, cols = 100, 90  # Not multiples of psize=32
        phase = np.random.rand(rows, cols) * 2 * np.pi - np.pi
        complex_data = np.exp(1j * phase).astype(np.complex64)

        result = goldstein(complex_data, alpha=0.5, psize=32)

        assert result.shape == complex_data.shape
        # Key regression test: edges should NOT be zero
        assert not np.allclose(result[-5:, :], 0), "Bottom edge should not be zero"
        assert not np.allclose(result[:, -5:], 0), "Right edge should not be zero"
        assert not np.allclose(
            result[-5:, -5:], 0
        ), "Bottom-right corner should not be zero"

    def test_goldstein_edge_normalization(self):
        """Test that edge regions have similar magnitude to interior."""
        np.random.seed(42)
        rows, cols = 100, 90
        phase = np.random.rand(rows, cols) * 2 * np.pi - np.pi
        complex_data = np.exp(1j * phase).astype(np.complex64)

        result = goldstein(complex_data, alpha=0.5, psize=32)

        interior_mag = np.abs(result[40:60, 40:50]).mean()
        edge_mag = np.abs(result[-10:, -10:]).mean()
        ratio = edge_mag / interior_mag
        # Edge magnitude should be within 10% of interior
        assert 0.9 < ratio < 1.1, f"Edge/interior ratio {ratio:.3f} not close to 1"

    def test_goldstein_preserves_zero_phase(self):
        """Test that data with phase=0 is not incorrectly marked as empty."""
        data = np.ones((64, 64), dtype=np.complex64)  # phase=0, magnitude=1
        result = goldstein(data, alpha=0.5, psize=32)
        # Should not be zeroed out
        assert not np.allclose(
            result, 0
        ), "Valid data with phase=0 should not be zeroed"

    def test_goldstein_small_array(self):
        """Test that arrays smaller than psize still work."""
        np.random.seed(42)
        small_data = np.exp(1j * np.random.rand(20, 25) * 2 * np.pi).astype(
            np.complex64
        )
        result = goldstein(small_data, alpha=0.5, psize=32)
        assert result.shape == small_data.shape
        assert not np.any(result == 0), "Small array should have no zeros"
