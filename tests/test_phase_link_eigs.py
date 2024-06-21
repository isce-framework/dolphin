import numpy as np
import numpy.testing as npt
import pytest

from dolphin.phase_link._eigenvalues import (
    eigh_largest_stack,
    eigh_smallest_stack,
)
from dolphin.phase_link.simulate import simulate_coh

# Used for matrix size
N = 20


def get_eigvec_phase_difference(a, b):
    """Compare the phase difference of two eigenvectors (or stacks).

    Using this since there may be a relative offset in the phase of
    each starting, depending on the algorithm.
    """
    a_0 = a * a[..., [0]].conj()
    b_0 = b * b[..., [0]].conj()
    return np.angle(a_0 * b_0.conj())


class TestEighStack:
    """Test the stack functionality to compare to scipy eigh."""

    @pytest.fixture
    def coh_stack(self):
        num_rows, num_cols = 6, 7
        out = np.empty((num_rows, num_cols, N, N), dtype=np.complex64)
        for row in range(num_rows):
            for col in range(num_cols):
                out[row, col] = simulate_coh(num_acq=N, add_signal=True)[0]
        return out

    @pytest.fixture
    def expected_largest(self, coh_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, -1]
        expected_evec = eig_vecs[:, :, :, -1]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N)
        return expected_eig, expected_evec

    def test_eigh_largest_stack(self, coh_stack, expected_largest):
        expected_eig, expected_evec = expected_largest
        evalues, evecs = eigh_largest_stack(coh_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)

        npt.assert_allclose(expected_eig, evalues, atol=2e-5)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 5e-3

    @pytest.fixture
    def coh_gamma_inv_stack(self, coh_stack) -> np.ndarray:
        return coh_stack * np.abs(np.linalg.inv(coh_stack))

    @pytest.fixture
    def expected_smallest(self, coh_gamma_inv_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_gamma_inv_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, 0]
        expected_evec = eig_vecs[:, :, :, 0]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N)
        return expected_eig, expected_evec

    def test_eigh_smallest_stack(self, coh_gamma_inv_stack, expected_smallest):
        expected_eig, expected_evec = expected_smallest
        mu = 0.99
        evalues, evecs = eigh_smallest_stack(coh_gamma_inv_stack, mu)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)

        # Compare the values:
        npt.assert_allclose(evalues, expected_eig, atol=2e-5)

        # Check the max phase difference
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4
