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
    ifg_filenames: List[Filename],
    cor_filenames: List[Filename],
    output_path: Filename,
    *,
    nlooks: float = 5,
    mask_file: Optional[Filename] = None,
    init_method: str = "mst",
    use_icu: bool = False,
    unw_suffix: str = ".unw.tif",
    max_jobs: int = 1,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[List[Path], List[Path]]:
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_filenames : list[Filename]
        Paths to input interferograms.
    cor_filenames : list[Filename]
        Paths to input correlation files. Order must match `ifg_filenames`.
    output_path : Filename
        Path to output directory.
    nlooks : int, optional
        Effective number of spatial looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst".
    use_icu : bool, optional
        Force the use of isce3's ICU instead of snaphu, by default False.
    unw_suffix : str, optional, default = ".unw.tif"
        unwrapped file suffix to use for creating/searching for existing files.
    max_jobs : int, optional, default = 4
        Maximum parallel processes.
    overwrite : bool, optional, default = False
        Overwrite existing unwrapped files.

    Returns
    -------
    unw_paths : list[Path]
        list of unwrapped files names
    conncomp_paths : list[Path]
        list of connected-component-label files names

    """
    if len(cor_filenames) != len(ifg_filenames):
        raise ValueError(
            "Number of correlation files does not match number of interferograms."
            f" Found {len(cor_filenames)} correlation files and"
            f" {len(ifg_filenames)} interferograms."
        )

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    output_path = Path(output_path)

    all_out_files = [
        (output_path / Path(f).name).with_suffix(unw_suffix) for f in ifg_filenames
    ]
    in_files, out_files = [], []
    for inf, outf in zip(ifg_filenames, all_out_files):
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

    # This keeps it from spawning a new process for a single job.
    Executor = ProcessPoolExecutor if max_jobs > 1 else DummyProcessPoolExecutor
    with Executor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                ifg_filename=ifg_file,
                corr_filename=cor_file,
                unw_filename=out_file,
                nlooks=nlooks,
                init_method=init_method,
                use_icu=use_icu,
                mask_file=mask_file,
            )
            for ifg_file, out_file, cor_file in zip(in_files, out_files, cor_filenames)
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
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = True,
    init_method: str = "mst",
    cost: str = "smooth",
    log_snaphu_to_file: bool = True,
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
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    log_snaphu_to_file : bool, optional
        Redirect SNAPHU's logging output to file, by default True
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

    shape = io.get_raster_xysize(ifg_filename)[::-1]
    corr_shape = io.get_raster_xysize(corr_filename)[::-1]
    if shape != corr_shape:
        raise ValueError(
            f"correlation {corr_shape} and interferogram {shape} shapes don't match"
        )
    mask_shape = io.get_raster_xysize(mask_file)[::-1] if mask_file else None
    if mask_file and shape != mask_shape:
        raise ValueError(
            f"Mask {mask_shape} and interferogram {shape} shapes don't match"
        )

    ifg_raster = Raster(fspath(ifg_filename))
    corr_raster = Raster(fspath(corr_filename))
    mask_raster = Raster(fspath(mask_file)) if mask_file else None
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

    if log_snaphu_to_file and use_snaphu:
        import journal

        logfile = Path(unw_filename).with_suffix(".log")
        journal.info("isce3.unwrap.snaphu").device = journal.logfile(
            fspath(logfile), "w"
        )
        logger.info(f"Logging snaphu output to {logfile}")

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        corr_raster = Raster(fspath(zeroed_corr_file))
        ifg_raster = Raster(fspath(zeroed_ifg_file))

    logger.info(
        f"Unwrapping size {(ifg_raster.length, ifg_raster.width)} {ifg_filename} to"
        f" {unw_filename} using {'SNAPHU' if use_snaphu else 'ICU'}"
    )
    if use_snaphu:
        snaphu.unwrap(
            unw_raster,
            conncomp_raster,
            ifg_raster,
            corr_raster,
            nlooks=nlooks,
            cost=cost,
            init_method=init_method,
            mask=mask_raster,
        )
    else:
        # Snaphu will fail on Mac OS due to a MemoryMap bug. Use ICU instead.
        # TODO: Should we zero out the correlation data using the mask,
        # since ICU doesn't support masking?

        icu = ICU(buffer_lines=shape[0])
        icu.unwrap(
            unw_raster,
            conncomp_raster,
            ifg_raster,
            corr_raster,
        )
    del unw_raster, conncomp_raster
    return Path(unw_filename), Path(conncomp_filename)


def _zero_from_mask(
    ifg_filename: Filename, corr_filename: Filename, mask_filename: Filename
) -> Tuple[Path, Path]:
    zeroed_ifg_file = Path(ifg_filename).with_suffix(".zeroed.tif")
    zeroed_corr_file = Path(corr_filename).with_suffix(".zeroed.cor.tif")

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
