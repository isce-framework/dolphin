from __future__ import annotations

from typing import Optional

from numpy.typing import ArrayLike

from ._ks import estimate_neighbors as estimate_neighbors_ks  # noqa: F401
from ._tf_test import estimate_neighbors as estimate_neighbors_tf  # noqa: F401


def estimate_neighbors(
    half_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    alpha: float,
    amp_stack: Optional[ArrayLike],
    mean: Optional[ArrayLike],
    var: Optional[ArrayLike],
    is_sorted: bool = False,
):
    """Estimate the statistically similar neighbors of each pixel."""
