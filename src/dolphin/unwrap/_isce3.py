from __future__ import annotations

import logging
from os import fspath
from pathlib import Path

import numpy as np

from dolphin import io
from dolphin._types import Filename
from dolphin.utils import full_suffix
from dolphin.workflows import UnwrapMethod

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA
from ._utils import _redirect_unwrapping_log, _zero_from_mask

logger = logging.getLogger(__name__)

__all__ = ["unwrap_isce3"]


def unwrap_isce3(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    mask_file: Filename | None = None,
    unwrap_method: UnwrapMethod = UnwrapMethod.PHASS,
    ccl_nodata: int = DEFAULT_CCL_NODATA,
    zero_where_masked: bool = False,
) -> tuple[Path, Path]:
    """Unwrap a single interferogram using isce3 or tophu.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    ccl_nodata : int, default = 65535
        Nodata value to use in output connected component raster
    unwrap_method : UnwrapMethod or str, optional, default = "phass"
        Choice of unwrapping algorithm to use.
        Choices: {"icu", "phass"} (snaphu is done by snaphu-py)

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    # Now we're using PHASS or ICU within isce3
    from isce3.io import Raster
    from isce3.unwrap import ICU, Phass

    shape = io.get_raster_xysize(ifg_filename)[::-1]
    corr_shape = io.get_raster_xysize(corr_filename)[::-1]
    if shape != corr_shape:
        msg = f"correlation {corr_shape} and interferogram {shape} shapes don't match"
        raise ValueError(msg)

    ifg_raster = Raster(fspath(ifg_filename))
    corr_raster = Raster(fspath(corr_filename))
    UNW_SUFFIX = full_suffix(unw_filename)

    # Get the driver based on the output file extension
    if Path(unw_filename).suffix == ".tif":
        driver = "GTiff"
        opts = list(io.DEFAULT_TIFF_OPTIONS)
    else:
        driver = "ENVI"
        opts = list(io.DEFAULT_ENVI_OPTIONS)

    unw_nodata = -10_000 if unwrap_method == UnwrapMethod.PHASS else 0
    # Create output rasters for unwrapped phase & connected component labels.
    # Writing with `io.write_arr` because isce3 doesn't have creation options
    io.write_arr(
        arr=None,
        output_name=unw_filename,
        driver=driver,
        like_filename=ifg_filename,
        dtype=np.float32,
        nodata=unw_nodata,
        options=opts,
    )
    conncomp_filename = str(unw_filename).replace(UNW_SUFFIX, CONNCOMP_SUFFIX)
    io.write_arr(
        arr=None,
        output_name=conncomp_filename,
        driver="GTiff",
        dtype=np.uint16,
        like_filename=ifg_filename,
        nodata=ccl_nodata,
        options=io.DEFAULT_TIFF_OPTIONS,
    )

    # The different raster classes have different APIs, so we need to
    # create the raster objects differently.
    unw_raster = Raster(fspath(unw_filename), True)
    conncomp_raster = Raster(fspath(conncomp_filename), True)

    _redirect_unwrapping_log(unw_filename, unwrap_method.value)

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        corr_raster = Raster(fspath(zeroed_corr_file))
        ifg_raster = Raster(fspath(zeroed_ifg_file))

    logger.info(
        f"Unwrapping size {(ifg_raster.length, ifg_raster.width)} {ifg_filename} to"
        f" {unw_filename} using {unwrap_method.value}"
    )
    if unwrap_method == UnwrapMethod.PHASS:
        # TODO: expose the configuration for phass?
        # If we ever find cases where changing help, then yes we should
        # coherence_thresh: float = 0.2
        # good_coherence: float = 0.7
        # min_region_size: int = 200
        unwrapper = Phass()
        unwrapper.unwrap(
            ifg_raster,
            corr_raster,
            unw_raster,
            conncomp_raster,
        )
    else:
        unwrapper = ICU(buffer_lines=shape[0])

        unwrapper.unwrap(
            unw_raster,
            conncomp_raster,
            ifg_raster,
            corr_raster,
        )
    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")
        return _zero_from_mask(unw_filename, conncomp_filename, mask_file)

    del unw_raster, conncomp_raster
    return Path(unw_filename), Path(conncomp_filename)
