from concurrent.futures import ProcessPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple

import isce3
import numpy as np
from isce3.unwrap import snaphu
from osgeo import gdal

from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.io import get_raster_xysize
from dolphin.utils import progress

logger = get_log(__name__)

gdal.UseExceptions()

# Input file extension
EXT_IFG = ".int"
# Output file extensions
EXT_CCL = ".conncomp.tif"
EXT_UNW = ".unw.tif"


@log_runtime
def run(
    ifg_path: Filename,
    output_path: Filename,
    cor_file: Filename,
    nlooks: float = 5,
    init_method: str = "mst",
    mask_file: Optional[Filename] = None,
    max_jobs: int = 4,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[List[Path], List[Path]]:
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_path : Filename
        Path to input interferograms
    output_path : Filename
        Path to output directory
    cor_file : str, optional
        location of (temporal) correlation file
    nlooks : int, optional
        Effective number of spatial looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to mask file, by default None
    max_jobs : int, optional
        Maximum parallel processes, by default 4
    overwrite : bool, optional
        overwrite results, by default False
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"

    Returns
    -------
    unw_paths : list[Path]
        list of unwrapped files names
    conncomp_paths : list[Path]
        list of connected-component-label files names

    """
    filenames = list(Path(ifg_path).glob("*" + EXT_IFG))
    if len(filenames) == 0:
        raise ValueError(
            f"No interferograms found in {ifg_path} with extension {EXT_IFG}"
        )

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    output_path = Path(output_path)

    all_out_files = [(output_path / f.name).with_suffix(EXT_UNW) for f in filenames]
    in_files, out_files = [], []
    for inf, outf in zip(filenames, all_out_files):
        if Path(outf).exists() and not overwrite:
            logger.info(f"{outf} exists. Skipping.")
            continue

        in_files.append(inf)
        out_files.append(outf)
    logger.info(f"{len(out_files)} left to unwrap")

    if mask_file:
        mask_file = Path(mask_file).resolve()
        # TODO: include mask_file in snaphu
        # Make sure it's the right format with 1s and 0s for include/exclude

    with ProcessPoolExecutor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                _run_isce3_snaphu,
                inf,
                cor_file,
                outf,
                nlooks,
                init_method,
            )
            for inf, outf in zip(in_files, out_files)
        ]
        with progress() as p:
            for fut in p.track(as_completed(futures), total=len(out_files)):
                fut.result()

    conncomp_files = [Path(outf).with_suffix(EXT_CCL) for outf in out_files]
    return all_out_files, conncomp_files


def _run_isce3_snaphu(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    init_method: str = "mst",
    cost: str = "smooth",
    log_to_file: bool = True,
) -> Tuple[Path, Path]:
    xsize, ysize = get_raster_xysize(ifg_filename)
    igram_raster = isce3.io.gdal.Raster(fspath(ifg_filename))
    corr_raster = isce3.io.gdal.Raster(fspath(corr_filename))

    # Create output rasters for unwrapped phase & connected component labels.
    driver = "GTiff"
    unw_raster = isce3.io.gdal.Raster(
        fspath(unw_filename), xsize, ysize, np.float32, driver
    )
    conncomp_filename = Path(unw_filename).with_suffix(EXT_CCL)
    conncomp_raster = isce3.io.gdal.Raster(
        fspath(conncomp_filename),
        xsize,
        ysize,
        np.uint32,
        driver,
    )
    if log_to_file:
        import journal

        logfile = Path(unw_filename).with_suffix(".log")
        logger.info(f"Unwrapping {unw_filename}: logging to {logfile}")
        journal.info("isce3.unwrap.snaphu").device = journal.logfile(
            fspath(logfile), "w"
        )

    snaphu.unwrap(
        unw_raster,
        conncomp_raster,
        igram_raster,
        corr_raster,
        nlooks=nlooks,
        cost=cost,
        init_method=init_method,
    )
    del unw_raster, conncomp_raster
    return Path(unw_filename), Path(conncomp_filename)
