from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np

from dolphin._types import Filename
from dolphin.io._core import DEFAULT_TIFF_OPTIONS_RIO
from dolphin.utils import full_suffix

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, DEFAULT_UNW_NODATA
from ._utils import _zero_from_mask

__all__ = [
    "unwrap_whirlwind",
]


logger = logging.getLogger("dolphin")


def unwrap_whirlwind(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = False,
    unw_nodata: Optional[float] = DEFAULT_UNW_NODATA,
    ccl_nodata: Optional[int] = DEFAULT_CCL_NODATA,
    interpolate: bool = False,
    interp_cutoff: float = 0.5,
    interp_num_neighbors: int = 20,
    interp_max_radius: int = 51,
    interp_min_radius: int = 0,
    interp_alpha: float = 0.75,
    conncomp_min_coherence: float | str | None = None,
    conncomp_reliability: float = 0.5,
    min_size_px: int = 100,
    max_ncomps: int = 1024,
    bridge: bool = True,
    connect_gaps: bool = False,
    connect_gaps_max_px: int = 300,
    goldstein_alpha: float = 0.0,
    goldstein_psize: int = 64,
) -> tuple[Path, Path]:
    """Unwrap an interferogram and grow conncomps using whirlwind.

    Uses ``whirlwind.unwrap``, which emits both the
    unwrapped phase and SNAPHU-style connected component labels from a
    single MCF solve.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file. Assumes 1 = valid, 0 = invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        Ignored if no mask is provided. Default False.
    unw_nodata : float, optional
        Nodata value for the output unwrapped phase raster.
    ccl_nodata : int, optional
        Nodata value for the connected component labels.
    interpolate : bool, optional
        Enable whirlwind's spiral PS interpolation pre-pass (fill valid pixels
        with coherence below ``interp_cutoff`` from nearby high-coherence
        phasors before unwrapping). Default False.
    interp_cutoff, interp_num_neighbors, interp_max_radius, interp_min_radius, \
interp_alpha
        Spiral interpolation parameters; see ``whirlwind.unwrap``. Only used
        when ``interpolate`` is True.
    conncomp_min_coherence : float or {"auto"} or None, optional
        Coherence floor for the component labels. None (the default) uses
        ``conncomp_reliability``; "auto" uses ww's looks-aware floor; a float
        in [0, 1] sets it explicitly. Default None.
    conncomp_reliability : float, optional
        Conservativeness of the component grow, in inverse-variance units: an
        edge becomes a component boundary when a one-cycle ambiguity flip
        across it costs no more than this. 0.5 is about a coherence-0.1 floor;
        0 labels nearly every unwrapped pixel. Used only when
        ``conncomp_min_coherence`` is None. Default 0.5.
    min_size_px : int, optional
        Discard connected components smaller than this many pixels. Default 100.
    max_ncomps : int, optional
        Maximum number of connected components to keep. Default 1024.
    bridge : bool, optional
        Bridge disjoint connected components across low-coherence gaps so they
        share a consistent integer cycle. Default True.
    connect_gaps : bool, optional
        Draw phase paths across bounded runs of invalid pixels before
        unwrapping, then drop those synthetic pixels from both outputs. The
        paths let the solver pick a relative 2pi level between regions that
        would otherwise be solved independently, which is what levels the
        NISAR sub-swaths. Unlike ``interpolate``, it extrapolates the phase
        slope on each side of a gap, so a winding estimate survives the
        crossing. Default False.
    connect_gaps_max_px : int, optional
        Widest invalid run, in pixels, that ``connect_gaps`` will cross.
        Default 300.
    goldstein_alpha : float, optional
        Strength of ww's internal Goldstein pre-filter. 0 disables it.
        Default 0.0.
    goldstein_psize : int, optional
        FFT patch size for ww's internal Goldstein filter. Default 64.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    import snaphu  # used here only for raster I/O
    import whirlwind as ww

    # Create a context manager that combines other context managers -- one for each
    # input raster file. Upon exiting the context block, each context manager in the
    # stack will be closed in LIFO order.
    with ExitStack() as stack:
        if zero_where_masked and (mask_file is not None):
            logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
            zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
                ifg_filename, corr_filename, mask_file
            )
            igram = stack.enter_context(snaphu.io.Raster(zeroed_ifg_file))
            corr = stack.enter_context(snaphu.io.Raster(zeroed_corr_file))
        else:
            igram = stack.enter_context(snaphu.io.Raster(ifg_filename))
            corr = stack.enter_context(snaphu.io.Raster(corr_filename))

        if mask_file is None:
            mask_arr = None
        else:
            mask = stack.enter_context(snaphu.io.Raster(mask_file))
            mask_arr = np.ascontiguousarray(mask[:, :], dtype=bool)

        logger.info("Unwrapping using whirlwind")
        igram_arr = np.ascontiguousarray(igram[:, :], dtype=np.complex64)
        corr_arr = np.ascontiguousarray(corr[:, :], dtype=np.float32)
        # ww.unwrap returns (phase, conncomp). As of whirlwind 2026-06-03 the
        # default phase solver is the verified single-tile linear MCF (ww-orig
        # parity + adaptive PD/SSP fallback);  Goldstein is off by default
        # (pass goldstein_alpha>0 to enable; under evaluation upstream).
        # Components always come from ww's SNAPHU-style grow with ThickenCosts
        # on, as production SNAPHU does. Pinned rather than left to ww's
        # defaults so a change upstream cannot silently alter dolphin's labels.
        unw, conncomp_arr = ww.unwrap(
            igram_arr,
            corr_arr,
            float(nlooks),
            mask=mask_arr,
            interpolate=interpolate,
            interp_cutoff=interp_cutoff,
            interp_num_neighbors=interp_num_neighbors,
            interp_max_radius=interp_max_radius,
            interp_min_radius=interp_min_radius,
            interp_alpha=interp_alpha,
            conncomp_algorithm="snaphu",
            conncomp_thicken=True,
            conncomp_min_coherence=conncomp_min_coherence,
            conncomp_reliability=conncomp_reliability,
            min_size_px=min_size_px,
            max_ncomps=max_ncomps,
            bridge=bridge,
            connect_gaps=connect_gaps,
            connect_gaps_max_px=connect_gaps_max_px,
            goldstein_alpha=goldstein_alpha,
            goldstein_psize=goldstein_psize,
        )

        logger.info("Writing unwrapped phase to raster file")
        with snaphu.io.Raster.create(
            unw_filename,
            like=igram,
            nodata=unw_nodata,
            dtype=np.float32,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as unw_raster:
            unw_raster[:, :] = unw

        unw_suffix = full_suffix(unw_filename)
        cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

        logger.info("Writing whirlwind connected component labels")
        with snaphu.io.Raster.create(
            cc_filename,
            like=igram,
            nodata=ccl_nodata,
            dtype=np.uint16,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as conncomp_raster:
            conncomp_raster[:, :] = conncomp_arr.astype(np.uint16)

    if zero_where_masked and (mask_file is not None):
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")
        return _zero_from_mask(unw_filename, cc_filename, mask_file)

    return Path(unw_filename), Path(cc_filename)
