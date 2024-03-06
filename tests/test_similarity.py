import numpy as np
import pytest

from dolphin import similarity


def test_get_circle_idxs():
    idxs = similarity.get_circle_idxs(3)
    expected = np.array(
        [
            [-2, -1],
            [-2, 0],
            [-2, 1],
            [-1, -2],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [0, -2],
            [0, -1],
            [0, 1],
            [0, 2],
            [1, -2],
            [1, -1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, -1],
            [2, 0],
            [2, 1],
        ]
    )
    np.testing.assert_array_equal(idxs, expected)


def test_pixel_similarity():
    x1, x2 = np.exp(1j * np.angle(np.random.randn(2, 50) + np.random.randn(2, 50) * 1j))
    sim = similarity.phase_similarity(x1[:5], x2[:5])
    assert -1 <= sim <= 1

    # For random noise, the similarity should be closer to zero with more data
    sim_full = similarity.phase_similarity(x1, x2)
    assert np.abs(sim_full) < np.abs(sim)

    # self similarty == 1
    assert similarity.phase_similarity(x1, x1) == 1


class TestMedianSimilarity:
    @pytest.fixture
    def ifg_stack(self, slc_stack):
        return slc_stack * slc_stack[[0]].conj()

    @pytest.mark.parametrize("radius", [2, 5, 9])
    def test_basic(self, ifg_stack, radius):
        sim = similarity.median_similarity(ifg_stack, search_radius=radius)
        assert np.all(sim >= -1)
        assert np.all(sim <= 1)

    @pytest.mark.parametrize("radius", [2, 5, 9])
    def test_median_similarity_weighted(self, ifg_stack, radius):
        rows, cols = ifg_stack.shape[-2:]
        weights = np.random.rand(rows, cols)
        sim = similarity.median_similarity(
            ifg_stack, search_radius=radius, weights=weights
        )
        assert np.all(sim >= -1)
        assert np.all(sim <= 1)
