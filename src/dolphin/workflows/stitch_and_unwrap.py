from pathlib import Path
from typing import List, Tuple

from dolphin import stitching, unwrap
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import VRTInterferogram

from .config import Workflow


@log_runtime
def run(
    ifg_list: List[VRTInterferogram],
    cfg: Workflow,
    debug: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_list : List[VRTInterferogram]
        List of [`VRTInterferogram`][dolphin.interferogram.VRTInterferogram] objects
        to stitch together
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
    stitched_ifg_dir = cfg.interferogram_network.directory / "stitched"
    stitched_ifg_dir.mkdir(exist_ok=True)

    # Also preps for snaphu, which needs binary format with no nans
    logger.info("Stitching interferograms by date.")
    ifg_filenames = [ifg.path for ifg in ifg_list]
    stitching.merge_by_date(
        image_file_list=ifg_filenames,  # type: ignore
        file_date_fmt=cfg.inputs.cslc_date_fmt,
        output_dir=stitched_ifg_dir,
    )

    # Stitch the correlation files
    pl_path = cfg.phase_linking.directory
    tcorr_files = list(pl_path.rglob("tcorr_average.tif"))
    stitched_cor_file = stitched_ifg_dir / "tcorr_average.tif"
    stitching.merge_images(
        tcorr_files,
        outfile=stitched_cor_file,
        driver="GTiff",
        overwrite=False,
    )

    # #####################################
    # 2. Unwrap the stitched interferograms
    # #####################################
    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return [], []

    logger.info(f"Unwrapping interferograms in {stitched_ifg_dir}")
    # Compute the looks for the unwrapping
    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks
    unwrapped_paths, conncomp_paths = unwrap.run(
        ifg_path=stitched_ifg_dir,
        output_path=cfg.unwrap_options.directory,
        cor_file=stitched_cor_file,
        nlooks=nlooks,
        # mask_file: Optional[Filename] = None,
        # TODO: max jobs based on the CPUs and the available RAM?
        # max_jobs=20,
        # overwrite: bool = False,
        no_tile=True,
    )

    # ####################
    # 3. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction

    return unwrapped_paths, conncomp_paths
