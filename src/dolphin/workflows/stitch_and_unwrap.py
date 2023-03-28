from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dolphin import io, stitching, unwrap
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import VRTInterferogram, estimate_correlation_from_phase

from .config import UnwrapMethod, Workflow


@log_runtime
def run(
    ifg_list: Sequence[VRTInterferogram],
    tcorr_file_list: Sequence[Path],
    cfg: Workflow,
    debug: bool = False,
) -> Tuple[List[Path], List[Path], Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_list : Sequence[VRTInterferogram]
        Sequence of [`VRTInterferogram`][dolphin.interferogram.VRTInterferogram] objects
        to stitch together
    tcorr_file_list : Sequence[Path]
        Sequence of paths to the correlation files for each interferogram
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
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
    ifg_filenames = [ifg.path for ifg in ifg_list]
    date_to_ifg_path = stitching.merge_by_date(
        image_file_list=ifg_filenames,  # type: ignore
        file_date_fmt=cfg.input_options.cslc_date_fmt,
        output_dir=stitched_ifg_dir,
    )

    # Estimate the spatial correlation from the stitched interferogram
    cor_paths = _estimate_spatial_correlations(
        date_to_ifg_path, window_size=cfg.phase_linking.half_window.to_looks()
    )

    # Stitch the correlation files
    stitched_cor_file = stitched_ifg_dir / "tcorr.tif"
    stitching.merge_images(
        tcorr_file_list,
        outfile=stitched_cor_file,
        driver="GTiff",
        overwrite=False,
    )

    # #####################################
    # 2. Unwrap the stitched interferograms
    # #####################################
    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return [], [], stitched_cor_file

    use_icu = cfg.unwrap_options.unwrap_method == UnwrapMethod.ICU
    # Note: ICU doesn't seem to support masks, but we'll zero the phase/cor
    logger.info(f"Unwrapping interferograms in {stitched_ifg_dir}")
    # Compute the looks for the unwrapping
    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks

    ifg_filenames = sorted(Path(stitched_ifg_dir).glob("*.int"))  # type: ignore
    if not ifg_filenames:
        raise FileNotFoundError(f"No interferograms found in {stitched_ifg_dir}")
    unwrapped_paths, conncomp_paths = unwrap.run(
        ifg_filenames=ifg_filenames,
        cor_filenames=cor_paths,
        output_path=cfg.unwrap_options._directory,
        nlooks=nlooks,
        mask_file=cfg.mask_file,
        # mask_file: Optional[Filename] = None,
        # TODO: max jobs based on the CPUs and the available RAM? use dask?
        # max_jobs=20,
        # overwrite: bool = False,
        no_tile=True,
        use_icu=use_icu,
    )

    # ####################
    # 3. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction

    return unwrapped_paths, conncomp_paths, stitched_cor_file


def _estimate_spatial_correlations(
    date_to_ifg_path: Dict[Tuple[date, ...], Path], window_size: Tuple[int, int]
) -> List[Path]:
    logger = get_log()

    cor_paths: List[Path] = []
    for dates, ifg_path in date_to_ifg_path.items():
        ifg = io.load_gdal(ifg_path)
        cor_path = ifg_path.with_suffix(".cor")
        cor_paths.append(cor_path)
        if cor_path.exists():
            logger.info(f"Skipping existing spatial correlation for {ifg_path}")
            continue
        logger.info(f"Estimating spatial correlation for {dates}...")
        cor = estimate_correlation_from_phase(ifg, window_size=window_size)
        logger.info(f"Writing spatial correlation to {cor_path}")
        io.write_arr(
            arr=cor,
            output_name=cor_path,
            like_filename=ifg_path,
            driver="ENVI",
            options=io.DEFAULT_ENVI_OPTIONS,
        )
    return cor_paths
