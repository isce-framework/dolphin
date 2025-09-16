import numpy as np
import pytest

from dolphin import similarity

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
)


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
    np.random.seed(12)
    phases1, phases2 = (
        np.random.uniform(2 * np.pi, size=(50,)),
        np.random.uniform(2 * np.pi, size=(50,)),
    )
    x1, x2 = np.exp(1j * phases1), np.exp(1j * phases2)
    small_sims = [similarity.phase_similarity(x1[:n], x2[:n]) for n in range(2, 6)]
    assert all(-1 <= sim <= 1 for sim in small_sims)

    # For random noise, the similarity should be closer to zero with more data
    sim_full = similarity.phase_similarity(x1, x2)
    assert np.abs(sim_full) < np.mean(np.abs(small_sims))

    # self similarity == 1
    assert similarity.phase_similarity(x1, x1) == 1


def test_pixel_similarity_zero_nan():
    x1, x2 = np.zeros((2, 10), dtype="complex64")
    sim = similarity.phase_similarity(x1, x2)
    assert sim == 0

    x1, x2 = np.full((2, 10), fill_value=np.nan + 1j * np.nan)
    sim = similarity.phase_similarity(x1, x2)
    assert np.isnan(sim)

    x1, x2 = np.ones((2, 10), dtype="complex64")
    sim = similarity.phase_similarity(x1, x2)
    assert sim == 1


def test_block_similarity_zero_nan():
    block_zeros = np.zeros((10, 4, 5), dtype="complex64")
    out = similarity.median_similarity(block_zeros, search_radius=2)
    assert np.isnan(out).all()

    block_nan = block_zeros * np.nan
    out = similarity.median_similarity(block_nan, search_radius=2)
    assert np.isnan(out).all()


class TestStackSimilarity:
    @pytest.fixture
    def ifg_stack(self, slc_stack):
        return slc_stack * slc_stack[[0]].conj()

    @pytest.mark.parametrize("radius", [2, 5, 9])
    @pytest.mark.parametrize("func", ["median", "max"])
    def test_basic(self, ifg_stack, radius, func):
        sim_func = getattr(similarity, f"{func}_similarity")
        sim = sim_func(ifg_stack, search_radius=radius)
        assert np.all(sim > -1)
        assert np.all(sim < 1)

    @pytest.mark.parametrize("radius", [2, 5, 9])
    @pytest.mark.parametrize("func", ["median", "max"])
    def test_max_similarity_masked(self, ifg_stack, radius, func):
        rows, cols = ifg_stack.shape[-2:]
        mask = np.random.rand(rows, cols).round().astype(bool)
        mask = np.array(
            [
                [True, False, False, True, False, False, False, False, True, True],
                [False, True, True, True, True, True, True, True, False, True],
                [True, False, False, False, False, True, True, False, True, False],
                [True, True, True, True, True, False, True, False, False, False],
                [False, False, True, True, True, True, True, True, False, True],
            ]
        )
        # Note: bottom right is surrounded by nans, so it will flip to a nan similarity

        sim_func = getattr(similarity, f"{func}_similarity")
        sim = sim_func(ifg_stack, search_radius=radius, mask=mask)

        assert ((np.nan_to_num(sim) > -1) & (np.nan_to_num(sim) < 1)).all()
        # The nan counts should be higher
        assert ~np.all(np.isnan(sim))
        assert np.isnan(sim).sum() >= (~mask).sum(), f"{sim = }, {mask = }"
        # The bottom right
        if radius == 2:
            assert np.isnan(sim[-1, -1])

    def test_create_similarity(self, tmp_path, slc_file_list):
        outfile = tmp_path / "med_sim.tif"
        similarity.create_similarities(
            slc_file_list, output_file=outfile, num_threads=1, block_shape=(64, 64)
        )
