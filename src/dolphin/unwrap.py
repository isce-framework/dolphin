from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import Optional, Sequence, Union

import isce3
import numpy as np
from isce3.unwrap import ICU, snaphu
from numba import njit, stencil
from osgeo import gdal

from dolphin import io
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.utils import full_suffix, progress
from dolphin.workflows import UnwrapMethod

logger = get_log(__name__)

gdal.UseExceptions()

CONNCOMP_SUFFIX = ".unw.conncomp"


@log_runtime
def run(
    ifg_filenames: Sequence[Filename],
    cor_filenames: Sequence[Filename],
    output_path: Filename,
    *,
    nlooks: float = 5,
    mask_file: Optional[Filename] = None,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    init_method: str = "mst",
    unw_suffix: str = ".unw.tif",
    max_jobs: int = 1,
    ntiles: Union[int, tuple[int, int]] = 1,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    scratchdir: Optional[Filename] = None,
    overwrite: bool = False,
    **kwargs,
) -> tuple[list[Path], list[Path]]:
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_filenames : Sequence[Filename]
        Paths to input interferograms.
    cor_filenames : Sequence[Filename]
        Paths to input correlation files. Order must match `ifg_filenames`.
    output_path : Filename
        Path to output directory.
    nlooks : int, optional
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    unwrap_method : UnwrapMethod or str, optional, default = "snaphu"
        Choice of unwrapping algorithm to use.
        Choices: {"snaphu", "icu", "phass"}
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst".
    unw_suffix : str, optional, default = ".unw.tif"
        unwrapped file suffix to use for creating/searching for existing files.
    max_jobs : int, optional, default = 1
        Maximum parallel processes.
    ntiles : int or tuple[int, int], optional, default = (1, 1)
        Use multi-resolution unwrapping with `tophu` on the interferograms.
        If 1 or (1, 1), doesn't use tophu and unwraps the interferogram as
        one single image.
    downsample_factor : int, optional, default = 1
        (For running coarse_unwrap): Downsample the interferograms by this
        factor to unwrap faster, then upsample to full resolution.
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
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
    Executor = ThreadPoolExecutor if max_jobs > 1 else DummyProcessPoolExecutor
    with Executor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                ifg_filename=ifg_file,
                corr_filename=cor_file,
                unw_filename=out_file,
                nlooks=nlooks,
                init_method=init_method,
                unwrap_method=unwrap_method,
                mask_file=mask_file,
                downsample_factor=downsample_factor,
                ntiles=ntiles,
                scratchdir=scratchdir,
            )
            for ifg_file, out_file, cor_file in zip(in_files, out_files, cor_filenames)
        ]
        with progress() as p:
            for fut in p.track(
                as_completed(futures),
                total=len(out_files),
                description="Unwrapping...",
                update_period=1,
            ):
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
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    init_method: str = "mst",
    cost: str = "smooth",
    log_to_file: bool = True,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    ntiles: Union[int, tuple[int, int]] = 1,
    scratchdir: Optional[Filename] = None,
) -> tuple[Path, Path]:
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
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    unwrap_method : UnwrapMethod or str, optional, default = "snaphu"
        Choice of unwrapping algorithm to use.
        Choices: {"snaphu", "icu", "phass"}
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    log_to_file : bool, optional
        Redirect SNAPHU's logging output to file, by default True
    downsample_factor : int, optional, default = 1
        Downsample the interferograms by this factor to unwrap faster, then upsample
        to full resolution.
        If 1, doesn't use coarse_unwrap and unwraps as normal.
    ntiles : int or tuple[int, int], optional, default = (1, 1)
        Use multi-resolution unwrapping with `tophu` on the interferograms.
        If 1 or (1, 1), doesn't use tophu and unwraps the interferogram as
        one single image.
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.

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
    if isinstance(downsample_factor, int):
        downsample_factor = (downsample_factor, downsample_factor)
    if isinstance(ntiles, int):
        ntiles = (ntiles, ntiles)
    # Coerce to the enum
    unwrap_method = UnwrapMethod(unwrap_method)

    if any(t > 1 for t in ntiles):
        return multiscale_unwrap(
            ifg_filename,
            corr_filename,
            unw_filename,
            downsample_factor,
            ntiles=ntiles,
            nlooks=nlooks,
            mask_file=mask_file,
            zero_where_masked=zero_where_masked,
            init_method=init_method,
            cost=cost,
            unwrap_method=unwrap_method,
            scratchdir=scratchdir,
            log_to_file=log_to_file,
        )

    # check not MacOS
    use_snaphu = sys.platform != "darwin" and unwrap_method not in ("icu", "phass")
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
) -> tuple[Path, Path]:
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


@njit(nogil=True)
def compute_phase_diffs(phase):
    """Compute the total number phase jumps > pi between adjacent pixels.

    If part of `phase` is known to be bad phase (e.g. over water),
    the values should be set to zero or a masked array should be passed:

        unwrapping_error(np.ma.masked_where(bad_area_mask, phase))


    Parameters
    ----------
    phase : ArrayLike
        Unwrapped interferogram phase.

    Returns
    -------
    int
        Total number of jumps exceeding pi.
    """
    return _compute_phase_diffs(phase)


@stencil
def _compute_phase_diffs(phase):
    d1 = np.abs(phase[0, 0] - phase[0, 1]) / np.pi
    d2 = np.abs(phase[0, 0] - phase[1, 0]) / np.pi
    # Subtract 0.5 so that anything below 1 gets rounded to 0
    return round(d1 - 0.5) + round(d2 - 0.5)


def _redirect_unwrapping_log(unw_filename: Filename, method: str):
    import journal

    logfile = Path(unw_filename).with_suffix(".log")
    journal.info(f"isce3.unwrap.{method}").device = journal.logfile(
        fspath(logfile), "w"
    )
    logger.info(f"Logging unwrapping output to {logfile}")


def multiscale_unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    downsample_factor: tuple[int, int],
    ntiles: tuple[int, int],
    nlooks: float,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = True,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    unwrap_callback=None,  # type is `tophu.UnwrapCallback`
    nodata: str | float | None = None,
    init_method: str = "mst",
    cost: str = "smooth",
    scratchdir: Optional[Filename] = None,
    log_to_file: bool = True,
) -> tuple[Path, Path]:
    """Unwrap an interferogram using at multiple scales using `tophu`.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    downsample_factor : tuple[int, int]
        Downsample the interferograms by this factor to unwrap faster, then upsample
    ntiles : tuple[int, int]
        Number of tiles to split for full image into for high res unwrapping.
        If `ntiles` is an int, will use `(ntiles, ntiles)`
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    unwrap_method : UnwrapMethod or str, optional, default = "snaphu"
        Choice of unwrapping algorithm to use.
        Choices: {"snaphu", "icu", "phass"}
    unwrap_callback : tophu.UnwrapCallback
        Alternative to `unwrap_method`: directly provide a callable
        function usable in `tophu`. See [tophu.UnwrapCallback] docs for interface.
    nodata : float | str, optional.
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
    log_to_file : bool, optional
        Redirect isce3's logging output to file, by default True

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.
    """
    import rasterio as rio
    import tophu

    def _get_rasterio_crs_transform(filename: Filename):
        with rio.open(filename) as src:
            return src.crs, src.transform

    def _get_cb_and_nodata(unwrap_method, unwrap_callback, nodata):
        if unwrap_callback is not None:
            # Pass through what the user gave
            return unwrap_callback, nodata
        unwrap_method = UnwrapMethod(unwrap_method)
        if unwrap_method == UnwrapMethod.ICU:
            unwrap_callback = tophu.ICUUnwrap()
            nodata = 0  # TODO: confirm this?
        elif unwrap_method == UnwrapMethod.PHASS:
            unwrap_callback = tophu.PhassUnwrap()
            nodata = -10_000
        elif unwrap_method == UnwrapMethod.SNAPHU:
            unwrap_callback = tophu.SnaphuUnwrap(
                cost=cost,
                init_method=init_method,
            )
            nodata = 0
        else:
            raise ValueError(f"Unknown {unwrap_method = }")
        return unwrap_callback, nodata

    # Used to track if we can redirect logs or not
    _user_gave_callback = unwrap_callback is not None
    unwrap_callback, nodata = _get_cb_and_nodata(unwrap_method, unwrap_callback, nodata)
    # Used to track if we can redirect logs or not

    (width, height) = io.get_raster_xysize(ifg_filename)
    crs, transform = _get_rasterio_crs_transform(ifg_filename)

    unw_suffix = full_suffix(unw_filename)
    conncomp_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

    # SUFFIX=ADD
    envi_options = dict(opt.lower().split("=") for opt in io.DEFAULT_ENVI_OPTIONS)
    logger.debug(f"Saving conncomps to {conncomp_filename}")
    conncomp_rb = tophu.RasterBand(
        conncomp_filename,
        height=height,
        width=width,
        dtype=np.uint16,
        driver="ENVI",
        crs=crs,
        transform=transform,
        **envi_options,
    )
    gtiff_options = dict(opt.lower().split("=") for opt in io.DEFAULT_TIFF_OPTIONS)
    unw_rb = tophu.RasterBand(
        unw_filename,
        height=height,
        width=width,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        **gtiff_options,
    )

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        igram_rb = tophu.RasterBand(zeroed_ifg_file)
        coherence_rb = tophu.RasterBand(zeroed_corr_file)
    else:
        igram_rb = tophu.RasterBand(ifg_filename)
        coherence_rb = tophu.RasterBand(corr_filename)

    if log_to_file and not _user_gave_callback:
        # Note that if they gave an arbitrary callback, we dont know the logger
        _redirect_unwrapping_log(unw_filename, unwrap_method.value)

    tophu.multiscale_unwrap(
        unw_rb,
        conncomp_rb,
        igram_rb,
        coherence_rb,
        nlooks=nlooks,
        unwrap_func=unwrap_callback,
        downsample_factor=downsample_factor,
        ntiles=ntiles,
        scratchdir=scratchdir,
    )

    return Path(unw_filename), Path(conncomp_filename)


def unwrap_snaphu_py(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    ntiles: tuple[int, int] = (1, 1),
    nproc: int = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = True,
    nodata: str | float | None = None,
    init_method: str = "mst",
) -> tuple[Path, Path]:
    """Unwrap an interferogram using at multiple scales using `tophu`.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    downsample_factor : tuple[int, int]
        Downsample the interferograms by this factor to unwrap faster, then upsample
    nlooks : float
        Effective number of looks used to form the input correlation data.
    ntiles : tuple[int, int], optional
        Number of (row, column) tiles to split for full image into.
        If `ntiles` is an int, will use `(ntiles, ntiles)`
    tile_overlap : tuple[int, int], optional
        Number of pixels to overlap in the (row, col) direction.
        Default = (0, 0)
    nproc : int, optional
        If specifying `ntiles`, number of processes to spawn to unwrap the
        tiles in parallel.
        Default = 1, which unwraps each tile in serial.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    nodata : float | str, optional.
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.
    """
    import snaphu

    unw_suffix = full_suffix(unw_filename)
    cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        igram = snaphu.io.Raster(zeroed_ifg_file)
        corr = snaphu.io.Raster(zeroed_corr_file)
    else:
        igram = snaphu.io.Raster(ifg_filename)
        corr = snaphu.io.Raster(corr_filename)

    mask = None if mask_file is None else snaphu.io.Raster(mask_file)
    try:
        with (
            snaphu.io.Raster.create(
                unw_filename, like=igram, nodata=nodata, dtype="f4"
            ) as unw,
            snaphu.io.Raster.create(
                cc_filename, like=igram, nodata=nodata, dtype="u4"
            ) as conncomp,
        ):
            # Unwrap and store the results in the `unw` and `conncomp` rasters.
            snaphu.unwrap(
                igram,
                corr,
                nlooks=nlooks,
                init=init_method,
                mask=mask,
                unw=unw,
                conncomp=conncomp,
                ntiles=ntiles,
                tile_overlap=tile_overlap,
                nproc=nproc,
            )
    finally:
        igram.close()
        corr.close()
        if mask is not None:
            mask.close()

    return Path(unw_filename), Path(cc_filename)
