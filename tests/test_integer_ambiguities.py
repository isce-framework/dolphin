"""Compare integer network estimates with exact small linear programs."""

import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import linprog

from dolphin._integer_ambiguities import _round_ambiguities
from dolphin.timeseries import (
    get_incidence_matrix,
    invert_stack_l1,
    invert_stack_l1_integer,
)


def exact_lad(A, b):
    m, n = A.shape
    result = linprog(
        np.r_[np.zeros(n), np.ones(2 * m)],
        A_eq=np.c_[A, np.eye(m), -np.eye(m)],
        b_eq=b,
        bounds=[(None, None)] * n + [(0, None)] * (2 * m),
        method="highs",
    )
    assert result.success
    return result.x[:n], result.fun


def graph(n, depth):
    return get_incidence_matrix(
        [(i, j) for i in range(n) for j in range(i + 1, min(n, i + depth + 1))]
    )


@pytest.mark.parametrize("n,depth", [(3, 2), (10, 1), (22, 3), (40, 2)])
def test_rounding_exact_optima_and_arbitrary_iterates(n, depth):
    rng = np.random.default_rng(145)
    A = graph(n, depth)
    for _ in range(12):
        b = rng.integers(-20, 21, len(A))
        optimum, objective = exact_lad(A, b)
        rounded, actual = _round_ambiguities(A, b, optimum)
        npt.assert_array_equal(rounded, np.rint(rounded))
        npt.assert_allclose(actual, objective, atol=1e-4)
        # Non-increase holds even away from a relaxed optimum.
        x = rng.uniform(-10, 10, n - 1)
        rounded, actual = _round_ambiguities(A, b, x)
        assert actual <= np.abs(A @ x - b).sum() + 1e-4
        npt.assert_allclose(actual, np.abs(A @ rounded - b).sum(), atol=1e-4)


@pytest.mark.parametrize(
    "x", [[0.25, 0.75], [-2.25, -1.25], [0.5, 0.5], [0, 1], [-1, -2]]
)
def test_shared_threshold_ties_negative_values_and_integers(x):
    A = graph(3, 2)
    b = np.array([0, 1, 0])
    x = np.array(x, dtype=float)
    points = np.unique(np.r_[0, 1 - x % 1, 1])
    shifts = (points[:-1] + points[1:]) / 2
    candidates = np.floor(x[:, None] + shifts)
    objectives = np.abs(A @ candidates - b[:, None]).sum(axis=0)
    rounded, objective = _round_ambiguities(A, b, x)
    assert np.any(np.all(candidates == np.asarray(rounded)[:, None], axis=0))
    npt.assert_allclose(objective, objectives.min())


def test_fractional_optimal_face():
    A = graph(3, 2)
    b = np.array([0, 1, 0])
    x = np.array([0.25, 0.75])
    npt.assert_allclose(np.abs(A @ x - b).sum(), 1)
    rounded, objective = _round_ambiguities(A, b, x)
    npt.assert_array_equal(rounded, np.rint(rounded))
    assert objective == 1


def test_random_edges_orientations_and_reference():
    rng = np.random.default_rng(824)
    pairs = [(i, i + 1) for i in range(11)]
    pairs += [tuple(rng.choice(12, 2, replace=False)) for _ in range(30)]
    full = get_incidence_matrix(pairs, delete_first_date_column=False)
    A = np.delete(full, 7, axis=1) * rng.choice([-1, 1], len(full))[:, None]
    b = rng.integers(-8, 9, len(A))
    optimum, objective = exact_lad(A, b)
    _, actual = _round_ambiguities(A, b, optimum)
    npt.assert_allclose(actual, objective, atol=1e-4)
    # Exercise validation on a connected graph with arbitrary edge orientation.
    _, actual = invert_stack_l1_integer(A, b[:, None, None], max_iter=200)
    npt.assert_allclose(actual, objective, atol=1e-4)


def test_unrepresentable_observations():
    if not jax.config.x64_enabled:
        with pytest.raises(ValueError, match="exactly representable"):
            invert_stack_l1_integer([[1]], [[[2**24 + 1]]])


def test_clean_phase_anchor():
    rng = np.random.default_rng(251)
    A = graph(22, 3)
    theta = rng.uniform(-np.pi, np.pi, (21, 3, 4))
    truth = rng.integers(-10, 11, theta.shape)
    phase = theta + 2 * np.pi * truth
    u = np.einsum("ij,jkl->ikl", A, phase)
    k = np.rint((u - np.einsum("ij,jkl->ikl", A, theta)) / (2 * np.pi))
    offsets, residuals = invert_stack_l1_integer(A, k)
    npt.assert_array_equal(offsets, truth)
    npt.assert_array_equal(residuals, np.zeros(theta.shape[1:]))
    # NumPy float64 reconstruction checks the convention independently of JAX.
    reconstructed = theta + 2 * np.pi * np.asarray(offsets, dtype=float)
    npt.assert_allclose(reconstructed, phase)
    npt.assert_allclose(np.angle(np.exp(1j * (reconstructed - theta))), 0, atol=1e-13)


def test_stack_objective_and_iteration_budget():
    rng = np.random.default_rng(20260912)
    A = graph(22, 3)
    truth = rng.integers(-20, 21, (21, 4, 5))
    errors = rng.integers(-3, 4, (60, 4, 5)) * (rng.random((60, 4, 5)) < 0.3)
    b = np.einsum("ij,jkl->ikl", A, truth) + errors
    for iterations in (1, 20, 200):
        relaxed, before = invert_stack_l1(A, b, max_iter=iterations)
        rounded, after = invert_stack_l1_integer(A, b, max_iter=iterations)
        assert rounded.shape == relaxed.shape == truth.shape
        assert after.shape == before.shape == (4, 5)
        npt.assert_array_equal(rounded, np.rint(rounded))
        assert np.all(after <= np.asarray(before) + 1e-3)
        npt.assert_allclose(
            after, np.abs(np.einsum("ij,jkl->ikl", A, rounded) - b).sum(axis=0)
        )
        if iterations == 200:
            for row, col in np.ndindex(after.shape):
                _, exact = exact_lad(A, b[:, row, col])
                npt.assert_allclose(after[row, col], exact, atol=1e-4)
    # A tiny ADMM budget can remain suboptimal: rounding is not a certificate.
    short, _ = invert_stack_l1_integer(A, b, max_iter=1)
    assert np.any(np.asarray(short) != np.asarray(rounded))


def test_dropped_edge_is_not_a_zero_observation():
    A = graph(3, 2)[[0, 2]]  # Observations 01 and 12 only.
    b = np.array([10, 10])[:, None, None]
    x, residual = invert_stack_l1_integer(A, b)
    npt.assert_array_equal(x[:, 0, 0], [10, 20])
    npt.assert_array_equal(residual, [[0]])


@pytest.mark.parametrize(
    "A,b,kwargs,message",
    [
        ([[1, 1]], [[[1]]], {}, "oriented graph edge"),
        ([[2]], [[[1]]], {}, "only -1, 0, and 1"),
        ([[-1, 1]], [[[1]]], {}, "connected"),
        ([[1]], [[[np.nan]]], {}, "finite integer"),
        ([[1]], [[[0.1]]], {}, "finite integer"),
        ([[1]], [[1]], {}, "shape"),
        ([[1]], [[[1]]], {"max_iter": 0}, "positive integer"),
        ([[1]], [[[1]]], {"max_iter": 1.5}, "positive integer"),
        (np.empty((0, 1)), np.empty((0, 1, 1)), {}, "nonempty"),
    ],
)
def test_invalid_inputs(A, b, kwargs, message):
    with pytest.raises(ValueError, match=message):
        invert_stack_l1_integer(A, b, **kwargs)


def test_masked_observation_is_rejected():
    b = np.ma.array([[[1]]], mask=True)
    with pytest.raises(ValueError, match="masked values"):
        invert_stack_l1_integer([[1]], b)


def test_rounding_is_jittable_and_vectorizable():
    A = graph(3, 2)
    b = np.array([[0, 1, 0], [0, 1, 0]]).T
    x = np.array([[0.25, 0.5], [0.75, 0.5]])
    rounded, objective = jax.jit(
        jax.vmap(_round_ambiguities, in_axes=(None, 1, 1), out_axes=(1, 0))
    )(A, b, x)
    assert rounded.shape == x.shape
    npt.assert_array_equal(objective, [1, 1])
