import jax.numpy as jnp
import numpy as np
import pytest
import troposim.igrams

from dolphin import timeseries


@pytest.fixture(scope="module")
def ifg_maker():
    igm = troposim.igrams.IgramMaker()
    igm.make_sar_stack()
    return igm


@pytest.fixture(scope="module")
def dphi(ifg_maker: troposim.igrams.IgramMaker):
    return ifg_maker.make_igram_stack(independent=False)


def test_incidence_matrix():
    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert A.shape == (5, 5)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
            [0, 0, 0, -1],
        ]
    )
    np.testing.assert_array_equal(A, expected)


def test_solve(ifg_maker: troposim.igrams.IgramMaker, dphi: jnp.ndarray):
    A = timeseries.get_incidence_matrix([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    phi = troposim.igrams.solve(A, dphi)
    assert phi.shape == (ifg_maker.num_sar_dates,)
