#!/usr/bin/env python
from typing import List

from dolphin import stitching, unwrap
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import VRTInterferogram

from .config import Workflow


@log_runtime
def run(ifg_list: List[VRTInterferogram], cfg: Workflow, debug: bool = False):
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

    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        return

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
    # TODO: Stitch the correlation files
    # tcorr_file = pl_path / "tcorr_average.tif"

    # #####################################
    # 2. Unwrap the stitched interferograms
    # #####################################

    logger.info(f"Unwrapping interferograms in {stitched_ifg_dir}")
    unwrapped_paths = unwrap.run(
        ifg_path=stitched_ifg_dir,
        output_path=cfg.unwrap_options.directory,
        cor_file=None,  # TODO: tcorr_file,
        # mask_file: Optional[Filename] = None,
        max_jobs=20,
        # overwrite: bool = False,
        no_tile=True,
    )

    # ####################
    # 3. Phase Corrections
    # ####################
    # TODO: Determine format for the tropospheric/ionospheric phase correction

    return unwrapped_paths
