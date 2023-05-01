from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from dolphin._log import get_log
from dolphin.workflows import ShpMethod

from . import _ks, _tf_test

logger = get_log(__name__)


def estimate_neighbors(
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    alpha: float,
    mean: Optional[ArrayLike],
    var: Optional[ArrayLike],
    nslc: int,
    amp_stack: Optional[ArrayLike],
    is_sorted: bool = False,
    shp_method: ShpMethod = ShpMethod.TF,
) -> Optional[np.ndarray]:
    """Estimate the statistically similar neighbors of each pixel."""
    if shp_method == ShpMethod.RECT:
        neighbor_arrays = None
    elif shp_method.lower() == ShpMethod.TF:
        if mean is None:
            mean = np.mean(amp_stack, axis=0)
        if var is None:
            var = np.var(amp_stack, axis=0)

        logger.info("Estimating SHP neighbors using KL distance")
        neighbor_arrays = _tf_test.estimate_neighbors(
            mean,
            var,
            halfwin_rowcol=halfwin_rowcol,
            nslc=nslc,
            alpha=alpha,
        )
    elif shp_method.lower() == ShpMethod.KS:
        if amp_stack is None:
            raise ValueError("amp_stack must be provided for KS method")
        logger.info("Estimating SHP neighbors using KS test")
        neighbor_arrays = _ks.estimate_neighbors(
            amp_stack,
            half_rowcol=halfwin_rowcol,
            strides_rowcol=strides_rowcol,
            alpha=alpha,
            is_sorted=is_sorted,
        )
    else:
        logger.warning(f"SHP method {shp_method} is not implemented for CPU yet")
        neighbor_arrays = None

    return neighbor_arrays
