from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

from dolphin import io, stitching, unwrap
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import estimate_correlation_from_phase

from .config import DisplacementWorkflow


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    tcorr_file_list: Sequence[Path],
    ps_file_list: Sequence[Path],
    cfg: DisplacementWorkflow,
    debug: bool = False,
    unwrap_jobs: int = 1,
) -> tuple[list[Path], list[Path], list[Path], Path, Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence of interferograms files.
        Separate bursts (if any) will be stitched together before unwrapping.
    tcorr_file_list : Sequence[Path]
        Sequence of paths to the burst-wise temporal coherence files.
    ps_file_list : Sequence[Path]
        Sequence of paths to the (looked) burst-wise ps mask files.
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    unwrap_jobs : int, default = 1
        Number of parallel unwrapping jobs to run at once.

    Returns
    -------
    unwrapped_paths : list[Path]
        list of Paths to unwrapped interferograms created.
    conncomp_paths : list[Path]
        list of Paths to connected component files created.
    interferometric_corr_paths : list[Path]
        list of Paths to interferometric correlation files created.
    stitched_tcorr_file : Path
        Path to temporal correlation file created.
    stitched_ps_file : Path
        Path to ps mask file created.
    """
    logger = get_log(debug=debug)

    # #########################################
    # 1. Stitch separate wrapped interferograms
    # #########################################

    # TODO: this should be made in the config
    stitched_ifg_dir = cfg.interferogram_network._directory / "stitched"
    stitched_ifg_dir.mkdir(exist_ok=True)

    # Also preps for snaphu, which needs binary format with no nans
    logger.info("Stitching interferograms by date.")
    date_to_ifg_path = stitching.merge_by_date(
        image_file_list=ifg_file_list,  # type: ignore
        file_date_fmt=cfg.input_options.cslc_date_fmt,
        output_dir=stitched_ifg_dir,
        out_bounds=cfg.output_options.bounds,
        out_bounds_epsg=cfg.output_options.bounds_epsg,
    )

    # Estimate the interferometric correlation from the stitched interferogram
    interferometric_corr_paths = _estimate_interferometric_correlations(
        date_to_ifg_path, window_size=cfg.phase_linking.half_window.to_looks()
    )

    # Stitch the correlation files
    stitched_tcorr_file = stitched_ifg_dir / "tcorr.tif"
    stitching.merge_images(
        tcorr_file_list,
        outfile=stitched_tcorr_file,
        driver="GTiff",
        out_bounds=cfg.output_options.bounds,
        out_bounds_epsg=cfg.output_options.bounds_epsg,
    )

    # Stitch the looked PS files
    stitched_ps_file = stitched_ifg_dir / "ps_mask_looked.tif"
    stitching.merge_images(
        ps_file_list,
        outfile=stitched_ps_file,
        out_nodata=255,
        driver="GTiff",
        resample_alg="nearest",
        out_bounds=cfg.output_options.bounds,
        out_bounds_epsg=cfg.output_options.bounds_epsg,
    )

    # #####################################
    # 2. Unwrap the stitched interferograms
    # #####################################
    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return [], [], [], stitched_tcorr_file, stitched_ps_file

    if cfg.mask_file is not None:
        # Check that the input mask is the same size as the ifgs:
        if io.get_raster_xysize(cfg.mask_file) == stitched_tcorr_file:
            logger.info(f"Using {cfg.mask_file} to mask during unwrapping")
            output_mask = cfg.mask_file
        else:
            logger.info(f"Warping {cfg.mask_file} to match size of interferograms")
            output_mask = stitched_ifg_dir / "warped_mask.tif"
            if output_mask.exists():
                logger.info(f"Mask already exists at {output_mask}")
            else:
                stitching.warp_to_match(
                    input_file=cfg.mask_file,
                    match_file=stitched_tcorr_file,
                    output_file=output_mask,
                )
    else:
        output_mask = None

    # Note: ICU doesn't seem to support masks, but we'll zero the phase/cor
    logger.info(f"Unwrapping interferograms in {stitched_ifg_dir}")
    # Compute the looks for the unwrapping
    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks

    ifg_filenames = sorted(Path(stitched_ifg_dir).glob("*.int"))  # type: ignore
    if not ifg_filenames:
        raise FileNotFoundError(f"No interferograms found in {stitched_ifg_dir}")

    # Make a scratch directory for unwrapping
    unwrap_scratchdir = cfg.unwrap_options._directory / "scratch"
    unwrap_scratchdir.mkdir(exist_ok=True)

    unwrapped_paths, conncomp_paths = unwrap.run(
        ifg_filenames=ifg_filenames,
        cor_filenames=interferometric_corr_paths,
        output_path=cfg.unwrap_options._directory,
        nlooks=nlooks,
        mask_file=output_mask,
        max_jobs=unwrap_jobs,
        ntiles=cfg.unwrap_options.ntiles,
        downsample_factor=cfg.unwrap_options.downsample_factor,
        unwrap_method=cfg.unwrap_options.unwrap_method,
        scratchdir=unwrap_scratchdir,
    )

    return (
        unwrapped_paths,
        conncomp_paths,
        interferometric_corr_paths,
        stitched_tcorr_file,
        stitched_ps_file,
    )


def _estimate_interferometric_correlations(
    date_to_ifg_path: dict[tuple[date, ...], Path], window_size: tuple[int, int]
) -> list[Path]:
    logger = get_log()

    corr_paths: list[Path] = []
    for dates, ifg_path in date_to_ifg_path.items():
        cor_path = ifg_path.with_suffix(".cor")
        corr_paths.append(cor_path)
        if cor_path.exists():
            logger.info(f"Skipping existing interferometric correlation for {ifg_path}")
            continue
        ifg = io.load_gdal(ifg_path)
        logger.info(f"Estimating interferometric correlation for {dates}...")
        cor = estimate_correlation_from_phase(ifg, window_size=window_size)
        logger.info(f"Writing interferometric correlation to {cor_path}")
        io.write_arr(
            arr=cor,
            output_name=cor_path,
            like_filename=ifg_path,
            driver="ENVI",
            options=io.DEFAULT_ENVI_OPTIONS,
        )
    return corr_paths
