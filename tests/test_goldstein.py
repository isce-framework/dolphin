import numpy as np
import pytest

from dolphin.goldstein import goldstein


@pytest.fixture
def smooth_complex():
    """A smooth, fully-valid complex interferogram patch."""
    ramp = np.linspace(0, 6, 64)[:, None] * np.ones((64, 64))
    return np.exp(1j * ramp).astype(np.complex64)


def test_output_shape_and_dtype_complex(smooth_complex):
    out = goldstein(smooth_complex, alpha=0.5, psize=32)
    assert out.shape == smooth_complex.shape
    assert np.iscomplexobj(out)


def test_real_phase_input_returns_complex(smooth_complex):
    """Real (float) phase input is wrapped to complex before filtering."""
    phase = np.angle(smooth_complex).astype(np.float64)
    out = goldstein(phase, alpha=0.5, psize=32)
    assert out.shape == phase.shape
    assert np.iscomplexobj(out)


def test_alpha_zero_is_near_identity(smooth_complex):
    """alpha=0 disables spectral weighting, so the overlap-add reconstructs input."""
    out = goldstein(smooth_complex, alpha=0.0, psize=32)
    np.testing.assert_allclose(out, smooth_complex, atol=1e-5)


def test_negative_alpha_raises(smooth_complex):
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        goldstein(smooth_complex, alpha=-0.1, psize=32)


def test_all_zero_returned_unchanged():
    data = np.zeros((40, 40), dtype=np.complex64)
    out = goldstein(data, alpha=0.5, psize=32)
    assert np.array_equal(out, data)


def test_all_nan_returned_unchanged():
    data = np.full((40, 40), np.nan, dtype=np.complex64)
    out = goldstein(data, alpha=0.5, psize=32)
    assert out.shape == data.shape
    assert np.all(np.isnan(out))


def test_empty_pixels_zeroed_in_output(smooth_complex):
    """Zero-magnitude (invalid) input pixels are forced back to 0 on output."""
    data = smooth_complex.copy()
    data[0, 0] = 0
    data[5, 7] = 0
    out = goldstein(data, alpha=0.5, psize=32)
    assert out[0, 0] == 0
    assert out[5, 7] == 0


def test_increasing_alpha_changes_output(smooth_complex):
    """Higher alpha applies stronger filtering, producing a different result."""
    out_low = goldstein(smooth_complex, alpha=0.1, psize=32)
    out_high = goldstein(smooth_complex, alpha=0.9, psize=32)
    assert not np.allclose(out_low, out_high)


def test_non_divisible_shape_preserved():
    """Input dims that are not a multiple of psize are padded and cropped back."""
    rng = np.random.default_rng(42)
    data = (rng.standard_normal((50, 37)) + 1j * rng.standard_normal((50, 37))).astype(
        np.complex64
    )
    out = goldstein(data, alpha=0.5, psize=32)
    assert out.shape == (50, 37)
    # Fully valid input -> no pixel forced to zero by the empty mask
    assert np.all(np.isfinite(out))
