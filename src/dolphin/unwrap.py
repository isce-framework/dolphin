import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import List, Optional, Tuple

import isce3
import numpy as np
from isce3.unwrap import ICU, snaphu
from osgeo import gdal

from dolphin import io
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.utils import full_suffix, progress

logger = get_log(__name__)

gdal.UseExceptions()

CONNCOMP_SUFFIX = ".unw.conncomp"


@log_runtime
def run(
    ifg_path: Filename,
    output_path: Filename,
    cor_file: Filename,
    nlooks: float = 5,
    init_method: str = "mst",
    mask_file: Optional[Filename] = None,
    ifg_suffix: str = ".int",
    unw_suffix: str = ".unw.tif",
    max_jobs: int = 1,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[List[Path], List[Path]]:
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_path : Filename
        Path to input interferograms.
    output_path : Filename
        Path to output directory.
    cor_file : str, optional
        location of (temporal) correlation file.
    nlooks : int, optional
        Effective number of spatial looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to mask file, by default None.
    ifg_suffix : str, optional, default = ".int"
        interferogram suffix to search for in `ifg_path`.
    unw_suffix : str, optional, default = ".unw.tif"
        unwrapped file suffix to use for creating/searching for existing files.
    max_jobs : int, optional, default = 4
        Maximum parallel processes.
    overwrite : bool, optional, default = False
        Overwrite existing unwrapped files.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst".

    Returns
    -------
    unw_paths : list[Path]
        list of unwrapped files names
    conncomp_paths : list[Path]
        list of connected-component-label files names

    """
    filenames = list(Path(ifg_path).glob("*" + ifg_suffix))
    if len(filenames) == 0:
        raise ValueError(
            f"No interferograms found in {ifg_path} with extension {ifg_suffix}"
        )

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    output_path = Path(output_path)

    all_out_files = [(output_path / f.name).with_suffix(unw_suffix) for f in filenames]
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

    # Don't even bother with the executor if there's only one job
    Executor = ProcessPoolExecutor if max_jobs > 1 else DummyProcessPoolExecutor
    with Executor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
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

    conncomp_files = [
        Path(str(outf).replace(unw_suffix, CONNCOMP_SUFFIX)) for outf in all_out_files
    ]
    return all_out_files, conncomp_files


def unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    init_method: str = "mst",
    cost: str = "smooth",
    log_to_file: bool = True,
    use_icu: bool = False,
) -> Tuple[Path, Path]:
    """Unwrap a single interferogram using isce3's SNAPHU/ICU bindings.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    nlooks : float
        Effective number of spatial looks used to form the input correlation data.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    log_to_file : bool, optional
        Log to file, by default True
    use_icu : bool, optional, default = False
        Force the unwrapping to use ICU

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    Notes
    -----
    On MacOS, the SNAPHU unwrapper doesn't work due to a MemoryMap bug.
    ICU is used instead.
    """
    # check not MacOS
    use_snaphu = sys.platform != "darwin" and not use_icu
    Raster = isce3.io.gdal.Raster if use_snaphu else isce3.io.Raster

    igram_raster = Raster(fspath(ifg_filename))
    corr_raster = Raster(fspath(corr_filename))

    unw_suffix = full_suffix(unw_filename)

    # Get the driver based on the output file extension
    if Path(unw_filename).suffix == ".tif":
        driver = "GTiff"
        opts = list(io.DEFAULT_TIFF_OPTIONS)
    else:
        driver = "ENVI"
        opts = list(io.DEFAULT_ENVI_OPTIONS)

    # Create output rasters for unwrapped phase & connected component labels.
    # Writing with `io.write_arr` because isce3 doesn't have creation options
    io.write_arr(
        arr=None,
        output_name=unw_filename,
        driver=driver,
        like_filename=ifg_filename,
        dtype=np.float32,
        options=opts,
    )
    # Always use ENVI for conncomp
    conncomp_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)
    io.write_arr(
        arr=None,
        output_name=conncomp_filename,
        driver="ENVI",
        dtype=np.uint32,
        like_filename=ifg_filename,
        options=io.DEFAULT_ENVI_OPTIONS,
    )
    if use_snaphu:
        unw_raster = Raster(fspath(unw_filename), 1, "w")
        conncomp_raster = Raster(fspath(conncomp_filename), 1, "w")
    else:
        # The different raster classes have different APIs, so we need to
        # create the raster objects differently.
        unw_raster = Raster(fspath(unw_filename), True)
        conncomp_raster = Raster(fspath(conncomp_filename), True)
    if log_to_file:
        import journal

        shape = (igram_raster.length, igram_raster.width)
        logfile = Path(unw_filename).with_suffix(".log")
        logger.info(
            f"Unwrapping {shape} {igram_raster} to {unw_filename}: logging to {logfile}"
        )
        journal.info("isce3.unwrap.snaphu").device = journal.logfile(
            fspath(logfile), "w"
        )

    if use_snaphu:
        snaphu.unwrap(
            unw_raster,
            conncomp_raster,
            igram_raster,
            corr_raster,
            nlooks=nlooks,
            cost=cost,
            init_method=init_method,
        )
    else:
        # Snaphu will fail on Mac OS due to a MemoryMap bug. Use ICU instead.
        icu = ICU()
        icu.unwrap(
            unw_raster,
            conncomp_raster,
            igram_raster,
            corr_raster,
        )
    del unw_raster, conncomp_raster
    return Path(unw_filename), Path(conncomp_filename)
