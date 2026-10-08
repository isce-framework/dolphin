"""Compare the existing LAD path with optional integer ambiguity rounding."""

import jax
import numpy as np

from dolphin.timeseries import (
    get_incidence_matrix,
    invert_stack_l1,
    invert_stack_l1_integer,
)


class IntegerLAD:
    params = ([22, 100], [False, True])
    param_names = ["dates", "integer"]

    def setup(self, dates, integer):
        rng = np.random.default_rng(20260913)
        pairs = [(i, j) for i in range(dates) for j in range(i + 1, min(dates, i + 4))]
        self.A = get_incidence_matrix(pairs)
        truth = rng.integers(-20, 21, (dates - 1, 64, 64))
        errors = rng.integers(-3, 4, (len(pairs), 64, 64))
        errors *= rng.random(errors.shape) < 0.1
        self.b = np.einsum("ij,jkl->ikl", self.A, truth) + errors
        self.solve = invert_stack_l1_integer if integer else invert_stack_l1
        jax.block_until_ready(self.solve(self.A, self.b))

    def time_invert(self, dates, integer):
        jax.block_until_ready(self.solve(self.A, self.b))
