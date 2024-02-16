from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Union

from tqdm.auto import tqdm

from dolphin import io
from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.utils import DummyProcessPoolExecutor, full_suffix
from dolphin.workflows import UnwrapMethod

from ._constants import CONNCOMP_SUFFIX, UNW_SUFFIX
from ._isce3 import unwrap_isce3
from ._snaphu_py import unwrap_snaphu_py
from ._tophu import multiscale_unwrap
from ._utils import create_combined_mask

logger = get_log(__name__)

__all__ = ["run", "unwrap"]


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
    cost: str = "smooth",
    max_jobs: int = 1,
    ntiles: Union[int, tuple[int, int]] = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    n_parallel_tiles: int = 1,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    scratchdir: Optional[Filename] = None,
    overwrite: bool = False,
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

    if mask_file:
        mask_file = Path(mask_file).resolve()

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
                mask_file=mask_file,
                downsample_factor=downsample_factor,
                ntiles=ntiles,
                tile_overlap=tile_overlap,
                n_parallel_tiles=n_parallel_tiles,
                scratchdir=scratchdir,
            )
            for ifg_file, out_file, cor_file in zip(in_files, out_files, cor_filenames)
        ]
        for fut in tqdm(as_completed(futures)):
            # We're not passing all the unw files in, so we need to tally up below
            _unw_path, _cc_path = fut.result()

    conncomp_files = [
        Path(str(outf).replace(UNW_SUFFIX, CONNCOMP_SUFFIX)) for outf in all_out_files
    ]
    return all_out_files, conncomp_files


def unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = False,
    ntiles: Union[int, tuple[int, int]] = 1,
    tile_overlap: tuple[int, int] = (0, 0),
    n_parallel_tiles: int = 1,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    init_method: str = "mst",
    cost: str = "smooth",
    log_to_file: bool = True,
    downsample_factor: Union[int, tuple[int, int]] = 1,
    scratchdir: Optional[Filename] = None,
) -> tuple[Path, Path]:
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
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    if isinstance(downsample_factor, int):
        downsample_factor = (downsample_factor, downsample_factor)
    if isinstance(ntiles, int):
        ntiles = (ntiles, ntiles)
    # Coerce to the enum
    unwrap_method = UnwrapMethod(unwrap_method)

    # Check for a nodata mask
    if io.get_raster_nodata(ifg_filename) is None or mask_file is None:
        # With no marked `nodata`, just use the passed in mask
        combined_mask_file = mask_file
    else:
        combined_mask_file = Path(ifg_filename).with_suffix(".mask.tif")
        create_combined_mask(
            mask_filename=mask_file,
            image_filename=ifg_filename,
            output_filename=combined_mask_file,
        )

    if unwrap_method == UnwrapMethod.SNAPHU:
        # Pass everything to snaphu-py
        unw_path, conncomp_path = unwrap_snaphu_py(
            ifg_filename,
            corr_filename,
            unw_filename,
            nlooks,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            mask_file=combined_mask_file,
            nproc=n_parallel_tiles,
            zero_where_masked=zero_where_masked,
            init_method=init_method,
            cost=cost,
            scratchdir=scratchdir,
        )
    elif any(t > 1 for t in ntiles):
        unw_path, conncomp_path = multiscale_unwrap(
            ifg_filename,
            corr_filename,
            unw_filename,
            downsample_factor,
            ntiles=ntiles,
            nlooks=nlooks,
            mask_file=combined_mask_file,
            zero_where_masked=zero_where_masked,
            init_method=init_method,
            cost=cost,
            unwrap_method=unwrap_method,
            scratchdir=scratchdir,
            log_to_file=log_to_file,
        )
    else:
        unw_path, conncomp_path = unwrap_isce3(
            ifg_filename,
            corr_filename,
            unw_filename,
            mask_file=combined_mask_file,
            unwrap_method=unwrap_method,
            zero_where_masked=zero_where_masked,
        )

    # TODO: post-processing steps go here:
    # Reset the input nodata values to be nodata in the `unw` and CCL
    return unw_path, conncomp_path
