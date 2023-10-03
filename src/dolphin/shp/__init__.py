from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from dolphin._log import get_log
from dolphin.workflows import ShpMethod

from . import _glrt, _kld, _ks

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
    method: ShpMethod = ShpMethod.GLRT,
    prune_disconnected: bool = False,
) -> np.ndarray:
    """Estimate the statistically similar neighbors of each pixel.

    Parameters
    ----------
    halfwin_rowcol : Tuple[int, int]
        Half window dimensions as a tuple (rows, columns).
    alpha : float
        Significance level (0 < alpha < 1).
    strides : dict[str, int], optional
        Strides for the x and y dimensions, by default {"x": 1, "y": 1}.
    mean : Optional[ArrayLike], optional
        Mean of the amplitude stack, by default None.
    var : Optional[ArrayLike], optional
        Variance of the amplitude stack, by default None.
    nslc : Optional[int], optional
        Number of samples, by default None.
    amp_stack : Optional[ArrayLike], optional
        Amplitude stack, by default None.
    is_sorted : bool, optional
        Whether the amplitude stack is sorted (if passed), by default False.
    method : ShpMethod, optional
        Method used for estimation, by default ShpMethod.GLRT.
    prune_disconnected : bool, default=False
        If True, keeps only SHPs that are 8-connected to the current pixel.
        Otherwise, any pixel within the window may be considered an SHP, even
        if it is not directly connected.

    Returns
    -------
    Optional[np.ndarray]
        Array of estimated statistically similar neighbors.

    Raises
    ------
    ValueError
        - nslc is not provided for GLRT/KLD methods
        - amp_stack is not provided for the KS method.
        - `method` not a valid `ShpMethod`
    """
    import numba

    logger.debug(f"NUMBA THREADS: {numba.get_num_threads()}")
    if method.lower() in (ShpMethod.KLD, ShpMethod.GLRT):
        if mean is None:
            mean = np.mean(amp_stack, axis=0)
        if var is None:
            var = np.var(amp_stack, axis=0)

    if method == ShpMethod.RECT:
        # No estimation needed
        neighbor_arrays = np.empty((0, 0, 0, 0), dtype=bool)
    elif method.lower() == ShpMethod.GLRT:
        logger.debug("Estimating SHP neighbors using GLRT")
        if nslc is None:
            raise ValueError("`nslc` must be provided for GLRT method")
        neighbor_arrays = _glrt.estimate_neighbors(
            mean=mean,
            var=var,
            halfwin_rowcol=halfwin_rowcol,
            strides=strides,
            nslc=nslc,
            alpha=alpha,
            prune_disconnected=prune_disconnected,
        )
    elif method.lower() == ShpMethod.KLD:
        logger.debug("Estimating SHP neighbors using KLD")
        if nslc is None:
            raise ValueError("`nslc` must be provided for GLRT method")
        neighbor_arrays = _kld.estimate_neighbors(
            mean=mean,
            var=var,
            halfwin_rowcol=halfwin_rowcol,
            strides=strides,
            nslc=nslc,
            alpha=alpha,
            prune_disconnected=prune_disconnected,
        )
    elif method.lower() == ShpMethod.KS:
        if amp_stack is None:
            raise ValueError("amp_stack must be provided for KS method")
        logger.debug("Estimating SHP neighbors using KS test")
        neighbor_arrays = _ks.estimate_neighbors(
            amp_stack=amp_stack,
            halfwin_rowcol=halfwin_rowcol,
            strides=strides,
            alpha=alpha,
            is_sorted=is_sorted,
        )
    else:
        raise ValueError(f"SHP method {method} is not implemented")

    return neighbor_arrays
