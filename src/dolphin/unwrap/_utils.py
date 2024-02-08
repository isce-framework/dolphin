from __future__ import annotations

from os import fspath
from pathlib import Path

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log(__name__)


def _zero_from_mask(
    ifg_filename: Filename, corr_filename: Filename, mask_filename: Filename
) -> tuple[Path, Path]:
    zeroed_ifg_file = Path(ifg_filename).with_suffix(".zeroed.tif")
    zeroed_corr_file = Path(corr_filename).with_suffix(".zeroed.cor.tif")

    if io.get_raster_xysize(ifg_filename) != io.get_raster_xysize(mask_filename):
        msg = f"Mask {mask_filename} and {ifg_filename} shapes don't match"
        raise ValueError(msg)

    mask = io.load_gdal(mask_filename)
    for in_f, out_f in zip(
        [ifg_filename, corr_filename], [zeroed_ifg_file, zeroed_corr_file]
    ):
        arr = io.load_gdal(in_f)
        arr[mask == 0] = 0
        logger.debug(f"Size: {arr.size}, {(arr != 0).sum()} non-zero pixels")
        io.write_arr(
            arr=arr,
            output_name=out_f,
            like_filename=corr_filename,
        )
    return zeroed_ifg_file, zeroed_corr_file


def _redirect_unwrapping_log(unw_filename: Filename, method: str):
    import journal

    logfile = Path(unw_filename).with_suffix(".log")
    journal.info(f"isce3.unwrap.{method}").device = journal.logfile(
        fspath(logfile), "w"
    )
    logger.info(f"Logging unwrapping output to {logfile}")
