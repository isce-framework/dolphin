from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from dolphin import goldstein, interpolate, io
from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.utils import DummyProcessPoolExecutor, full_suffix
from dolphin.workflows import UnwrapMethod

from ._constants import (
    CONNCOMP_SUFFIX,
    CONNCOMP_SUFFIX_ZEROED,
    DEFAULT_CCL_NODATA,
    DEFAULT_UNW_NODATA,
    UNW_SUFFIX,
    UNW_SUFFIX_ZEROED,
)
from ._snaphu_py import grow_conncomp_snaphu, unwrap_snaphu_py
from ._tophu import multiscale_unwrap
from ._utils import create_combined_mask, set_nodata_values

logger = get_log(__name__)

__all__ = ["run", "unwrap"]


@log_runtime
def run(
    ifg_filenames: Sequence[Filename],
    cor_filenames: Sequence[Filename],
    output_path: Filename,
    *,
    nlooks: float = 5,
    mask_filename: Optional[Filename] = None,
    zero_where_masked: bool = False,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    init_method: str = "mst",
    cost: str = "smooth",
    max_jobs: int = 1,
    ntiles: Union[int, tuple[int, int]] = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    n_parallel_tiles: int = 1,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    unw_nodata: float | None = DEFAULT_UNW_NODATA,
    ccl_nodata: int | None = DEFAULT_CCL_NODATA,
    scratchdir: Optional[Filename] = None,
    overwrite: bool = False,
    run_goldstein: bool = False,
    alpha: float = 0.5,
    run_interpolation: bool = False,
    max_radius: int = 51,
    interpolation_cor_threshold: float = 0.5,
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
    mask_filename : Filename, optional
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
        SNAPHU initialization method, by default "mst".
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    max_jobs : int, optional, default = 1
        Maximum parallel processes.
    ntiles : int or tuple[int, int], optional, default = (1, 1)
        Use multi-resolution unwrapping with `tophu` on the interferograms.
        If 1 or (1, 1), doesn't use tophu and unwraps the interferogram as
        one single image.
    tile_overlap : tuple[int, int], optional
        (For snaphu-py tiling): Number of pixels to overlap in the (row, col) direction.
        Default = (0, 0)
    n_parallel_tiles : int, optional
        (For snaphu-py tiling) If specifying `ntiles`, number of tiles to unwrap
        in parallel for each interferogram.
        Default = 1, which unwraps each tile in serial.
    downsample_factor : int, optional, default = 1
        (For tophu/multi-scale unwrapping): Downsample the interferograms by this
        factor to unwrap faster, then upsample to full resolution.
    unw_nodata : float , optional.
        Requested nodata value for the unwrapped phase.
        Default = 0
    ccl_nodata : float, optional
        Requested nodata value for connected component labels.
        Default = max value of UInt16 (65535)
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
    overwrite : bool, optional, default = False
        Overwrite existing unwrapped files.
    run_goldstein : bool, optional, default = False
        Whether to run Goldstein filtering on interferogram
    alpha : float, optional, default = 0.5
        Alpha parameter for Goldstein filtering
    run_interpolation : bool, optional, default = False
        Whether to run interpolation on interferogram
    max_radius : int, optional, default = 51
        maximum radius (in pixel) for scatterer searching for interpolation
    interpolation_cor_threshold : float, optional, default = 0.5
        Threshold on the correlation raster to use for interpolation.
        Pixels with less than this value are replaced by a weighted
        combination of neighboring pixels.

    Returns
    -------
    unw_paths : list[Path]
        list of unwrapped files names
    conncomp_paths : list[Path]
        list of connected-component-label files names

    """
    if len(cor_filenames) != len(ifg_filenames):
        msg = (
            "Number of correlation files does not match number of interferograms."
            f" Found {len(cor_filenames)} correlation files and"
            f" {len(ifg_filenames)} interferograms."
        )
        raise ValueError(msg)

    if init_method.lower() not in ("mcf", "mst"):
        msg = f"Invalid init_method {init_method}"
        raise ValueError(msg)

    output_path = Path(output_path)

    ifg_suffixes = [full_suffix(f) for f in ifg_filenames]
    all_out_files = [
        (output_path / Path(f).name.replace(suf, UNW_SUFFIX))
        for f, suf in zip(ifg_filenames, ifg_suffixes)
    ]
    in_files, out_files = [], []
    for inf, outf in zip(ifg_filenames, all_out_files):
        if Path(outf).exists() and not overwrite:
            logger.info(f"{outf} exists. Skipping.")
            continue

        in_files.append(inf)
        out_files.append(outf)
    logger.info(f"{len(out_files)} left to unwrap")

    if mask_filename:
        mask_filename = Path(mask_filename).resolve()

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
                cost=cost,
                unwrap_method=unwrap_method,
                mask_filename=mask_filename,
                zero_where_masked=zero_where_masked,
                downsample_factor=downsample_factor,
                ntiles=ntiles,
                tile_overlap=tile_overlap,
                n_parallel_tiles=n_parallel_tiles,
                unw_nodata=unw_nodata,
                ccl_nodata=ccl_nodata,
                scratchdir=scratchdir,
                run_goldstein=run_goldstein,
                alpha=alpha,
                run_interpolation=run_interpolation,
                max_radius=max_radius,
                interpolation_cor_threshold=interpolation_cor_threshold,
            )
            for ifg_file, out_file, cor_file in zip(in_files, out_files, cor_filenames)
        ]
        for fut in tqdm(as_completed(futures)):
            # We're not passing all the unw files in, so we need to tally up below
            _unw_path, _cc_path = fut.result()

    if zero_where_masked and mask_filename is not None:
        all_out_files = [
            Path(str(outf).replace(UNW_SUFFIX, UNW_SUFFIX_ZEROED))
            for outf in all_out_files
        ]
        conncomp_files = [
            Path(str(outf).replace(UNW_SUFFIX_ZEROED, CONNCOMP_SUFFIX_ZEROED))
            for outf in all_out_files
        ]
    else:
        conncomp_files = [
            Path(str(outf).replace(UNW_SUFFIX, CONNCOMP_SUFFIX))
            for outf in all_out_files
        ]
    return all_out_files, conncomp_files


def unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_filename: Optional[Filename] = None,
    zero_where_masked: bool = False,
    ntiles: Union[int, tuple[int, int]] = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    n_parallel_tiles: int = 1,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    init_method: str = "mst",
    cost: str = "smooth",
    log_to_file: bool = True,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    unw_nodata: float | None = DEFAULT_UNW_NODATA,
    ccl_nodata: int | None = DEFAULT_CCL_NODATA,
    scratchdir: Optional[Filename] = None,
    run_goldstein: bool = False,
    alpha: float = 0.5,
    run_interpolation: bool = False,
    max_radius: int = 51,
    interpolation_cor_threshold: float = 0.5,
) -> tuple[Filename, Filename]:
    """Unwrap a single interferogram using snaphu, isce3, or tophu.

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
    mask_filename : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    unwrap_method : UnwrapMethod or str, optional, default = "snaphu"
        Choice of unwrapping algorithm to use.
        Choices: {"snaphu", "icu", "phass"}
    ntiles : int or tuple[int, int], optional, default = (1, 1)
        For either snaphu-py or tophu: divide the interferogram into tiles
        and unwrap each separately, then combine.
        If 1 or (1, 1), no tiling is performed, unwraps the interferogram as
        one single image.
    tile_overlap : tuple[int, int], optional
        (For snaphu-py tiling): Number of pixels to overlap in the (row, col) direction.
        Default = (0, 0)
    n_parallel_tiles : int, optional
        (For snaphu-py tiling) If specifying `ntiles`, number of processes to spawn
        to unwrap the tiles in parallel.
        Default = 1, which unwraps each tile in serial.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    log_to_file : bool, optional
        Redirect isce3 logging output to file, by default True
    downsample_factor : int, optional, default = 1
        Downsample the interferograms by this factor to unwrap faster, then upsample
        to full resolution.
        If 1, doesn't use coarse_unwrap and unwraps as normal.
    unw_nodata : float , optional.
        Requested nodata value for the unwrapped phase.
        Default = 0
    ccl_nodata : float, optional
        Requested nodata value for connected component labels.
        Default = max value of UInt16 (65535)
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
    run_goldstein : bool, optional, default = False
        Whether to run Goldstein filtering on interferogram
    alpha : float, optional, default = 0.5
        Alpha parameter for Goldstein filtering
    run_interpolation : bool, optional, default = False
        Whether to run interpolation on interferogram
    max_radius : int, optional, default = 51
        maximum radius (in pixel) for scatterer searching for interpolation
    interpolation_cor_threshold : float, optional, default = 0.5
        Threshold on the correlation raster to use for interpolation.
        Pixels with less than this value are replaced by a weighted
        combination of neighboring pixels.

    Returns
    -------
    unw_path : Filename
        Path to output unwrapped phase file.
    conncomp_path : Filename
        Path to output connected component label file.

    """
    if isinstance(downsample_factor, int):
        downsample_factor = (downsample_factor, downsample_factor)
    if isinstance(ntiles, int):
        ntiles = (ntiles, ntiles)
    # Coerce to the enum
    unwrap_method = UnwrapMethod(unwrap_method)

    # Check for a nodata mask
    if io.get_raster_nodata(ifg_filename) is None or mask_filename is None:
        # With no marked `nodata`, just use the passed in mask
        combined_mask_file = mask_filename
    else:
        combined_mask_file = Path(ifg_filename).with_suffix(".mask.tif")
        create_combined_mask(
            mask_filename=mask_filename,
            image_filename=ifg_filename,
            output_filename=combined_mask_file,
        )

    unwrapper_ifg_filename = Path(ifg_filename)
    unwrapper_unw_filename = Path(unw_filename)
    name_change = "."

    if run_goldstein:
        suf = Path(unw_filename).suffix
        if suf == ".tif":
            driver = "GTiff"
            opts = list(io.DEFAULT_TIFF_OPTIONS)
        else:
            driver = "ENVI"
            opts = list(io.DEFAULT_ENVI_OPTIONS)

        name_change = ".filt" + name_change
        # If we're running Goldstein filtering, the intermediate
        # filtered/unwrapped rasters are temporary rasters in the scratch dir.
        filt_ifg_filename = Path(scratchdir or ".") / (
            Path(ifg_filename).stem.split(".")[0] + (name_change + "int" + suf)
        )
        filt_unw_filename = Path(
            str(unw_filename).split(".")[0] + (name_change + "unw" + suf)
        )

        ifg = io.load_gdal(ifg_filename)
        logger.info(f"Goldstein filtering {ifg_filename} -> {filt_ifg_filename}")
        modified_ifg = goldstein(ifg, alpha=alpha)
        logger.info(f"Writing filtered output to {filt_ifg_filename}")
        io.write_arr(
            arr=modified_ifg,
            output_name=filt_ifg_filename,
            like_filename=ifg_filename,
            driver=driver,
            options=opts,
        )
        unwrapper_ifg_filename = filt_ifg_filename
        unwrapper_unw_filename = filt_unw_filename

    if run_interpolation:
        suf = Path(ifg_filename).suffix
        if suf == ".tif":
            driver = "GTiff"
            opts = list(io.DEFAULT_TIFF_OPTIONS)
        else:
            driver = "ENVI"
            opts = list(io.DEFAULT_ENVI_OPTIONS)

        pre_interp_ifg_filename = unwrapper_ifg_filename
        pre_interp_unw_filename = unwrapper_unw_filename
        name_change = ".interp" + name_change

        # temporarily storing the intermediate interpolated rasters in the scratch dir.
        interp_ifg_filename = Path(scratchdir or ".") / (
            pre_interp_ifg_filename.stem.split(".")[0] + (name_change + "int" + suf)
        )
        interp_unw_filename = Path(
            str(pre_interp_unw_filename).split(".")[0] + (name_change + "unw" + suf)
        )

        ifg = io.load_gdal(pre_interp_ifg_filename)
        corr = io.load_gdal(corr_filename)
        logger.info(
            f"Masking pixels with correlation below {interpolation_cor_threshold}"
        )
        coherent_pixel_mask = corr[:] >= interpolation_cor_threshold

        logger.info(f"Interpolating {pre_interp_ifg_filename} -> {interp_ifg_filename}")
        modified_ifg = interpolate(
            ifg=ifg,
            weights=coherent_pixel_mask,
            weight_cutoff=interpolation_cor_threshold,
            max_radius=max_radius,
        )

        logger.info(f"Writing interpolated output to {interp_ifg_filename}")
        io.write_arr(
            arr=modified_ifg,
            output_name=interp_ifg_filename,
            like_filename=ifg_filename,
            driver=driver,
            options=opts,
        )
        unwrapper_ifg_filename = interp_ifg_filename
        unwrapper_unw_filename = interp_unw_filename

    if unwrap_method == UnwrapMethod.SNAPHU:
        # Pass everything to snaphu-py
        unw_path, conncomp_path = unwrap_snaphu_py(
            unwrapper_ifg_filename,
            corr_filename,
            unwrapper_unw_filename,
            nlooks,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            mask_file=combined_mask_file,
            nproc=n_parallel_tiles,
            zero_where_masked=zero_where_masked,
            unw_nodata=unw_nodata,
            ccl_nodata=ccl_nodata,
            init_method=init_method,
            cost=cost,
            scratchdir=scratchdir,
        )
    else:
        unw_path, conncomp_path = multiscale_unwrap(
            unwrapper_ifg_filename,
            corr_filename,
            unwrapper_unw_filename,
            downsample_factor,
            ntiles=ntiles,
            nlooks=nlooks,
            mask_file=combined_mask_file,
            zero_where_masked=zero_where_masked,
            unw_nodata=unw_nodata,
            ccl_nodata=ccl_nodata,
            init_method=init_method,
            cost=cost,
            unwrap_method=unwrap_method,
            scratchdir=scratchdir,
            log_to_file=log_to_file,
        )

    # TODO: post-processing steps go here:

    # Reset the input nodata values to be nodata in the unwrapped and CCL
    logger.info(f"Setting nodata values of {unw_path} file")
    set_nodata_values(
        filename=unw_path, output_nodata=unw_nodata, like_filename=ifg_filename
    )
    logger.info(f"Setting nodata values of {conncomp_path} file")
    set_nodata_values(
        filename=conncomp_path, output_nodata=ccl_nodata, like_filename=ifg_filename
    )

    # Transfer ambiguity numbers from filtered/interpolated unwrapped interferogram
    # back to original interferogram
    if run_goldstein or run_interpolation:
        logger.info(
            "Transferring ambiguity numbers from filtered/interpolated"
            "ifg {scratch_unw_filename}"
        )
        unw_arr = io.load_gdal(unwrapper_unw_filename)

        final_arr = np.angle(ifg) + (unw_arr - np.angle(modified_ifg))

        io.write_arr(
            arr=final_arr,
            output_name=unw_filename,
            dtype=np.float32,
            driver=driver,
            options=opts,
        )

        # Regrow connected components after phase modification
        corr = io.load_gdal(corr_filename)
        mask = corr[:] > 0
        # TODO decide whether we want to have the
        # 'min_conncomp_frac' option in the config
        conncomp_path = grow_conncomp_snaphu(
            unw_filename=unw_filename,
            corr_filename=corr_filename,
            nlooks=nlooks,
            mask=mask,
            ccl_nodata=ccl_nodata,
            cost=cost,
            scratchdir=scratchdir,
        )

    return unw_path, conncomp_path
