from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from dolphin._log import get_log
from dolphin.workflows import ShpMethod

from . import _ks, _tf_test

logger = get_log(__name__)

__all__ = ["estimate_neighbors"]


def estimate_neighbors(
    *,
    halfwin_rowcol: tuple[int, int],
    alpha: float,
    strides: dict[str, int] = {"x": 1, "y": 1},
    mean: Optional[ArrayLike] = None,
    var: Optional[ArrayLike] = None,
    nslc: Optional[int] = None,
    amp_stack: Optional[ArrayLike] = None,
    is_sorted: bool = False,
    method: ShpMethod = ShpMethod.TF,
) -> Optional[np.ndarray]:
    """Estimate the statistically similar neighbors of each pixel."""
    if method == ShpMethod.RECT:
        neighbor_arrays = None
    elif method.lower() == ShpMethod.TF:
        if nslc is None:
            raise ValueError("`nslc` must be provided for TF method")
        if mean is None:
            mean = np.mean(amp_stack, axis=0)
        if var is None:
            var = np.var(amp_stack, axis=0)

        logger.debug("Estimating SHP neighbors using T- and F-test")
        neighbor_arrays = _tf_test.estimate_neighbors(
            mean,
            var,
            halfwin_rowcol=halfwin_rowcol,
            strides=strides,
            nslc=nslc,
            alpha=alpha,
        )
    elif method.lower() == ShpMethod.KS:
        if amp_stack is None:
            raise ValueError("amp_stack must be provided for KS method")
        logger.debug("Estimating SHP neighbors using KS test")
        neighbor_arrays = _ks.estimate_neighbors(
            amp_stack,
            halfwin_rowcol=halfwin_rowcol,
            strides=strides,
            alpha=alpha,
            is_sorted=is_sorted,
        )
    else:
        logger.warning(f"SHP method {method} is not implemented for CPU yet")
        neighbor_arrays = None

    return neighbor_arrays
