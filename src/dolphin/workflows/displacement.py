#!/usr/bin/env python
from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pformat

from opera_utils import group_by_burst, group_by_date

from dolphin import __version__
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin.atmosphere import estimate_ionospheric_delay, estimate_tropospheric_delay
from dolphin.utils import get_max_memory_usage, set_num_threads

from . import stitching_bursts, unwrapping, wrapped_phase
from ._utils import _create_burst_cfg, _remove_dir_if_empty
from .config import DisplacementWorkflow

logger = get_log(__name__)


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    debug: bool = False,
):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    """
    # Set the logging level for all `dolphin.` modules
    logger = get_log(name="dolphin", debug=debug, filename=cfg.log_file)
    logger.debug(pformat(cfg.model_dump()))

    set_num_threads(cfg.worker_settings.threads_per_worker)

    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise e
        # Otherwise, we have SLC files which are not OPERA burst files
        grouped_slc_files = {"": cfg.cslc_file_list}

    if cfg.amplitude_dispersion_files:
        grouped_amp_dispersion_files = group_by_burst(cfg.amplitude_dispersion_files)
    else:
        grouped_amp_dispersion_files = defaultdict(list)
    if cfg.amplitude_mean_files:
        grouped_amp_mean_files = group_by_burst(cfg.amplitude_mean_files)
    else:
        grouped_amp_mean_files = defaultdict(list)

    if len(cfg.correction_options.troposphere_files) > 0:
        grouped_tropo_files = group_by_date(
            cfg.correction_options.troposphere_files,
            file_date_fmt=cfg.correction_options.tropo_date_fmt,
        )
    else:
        grouped_tropo_files = None

    if len(cfg.correction_options.ionosphere_files) > 0:
        grouped_iono_files = group_by_date(
            cfg.correction_options.ionosphere_files,
            file_date_fmt=cfg.correction_options.iono_date_fmt,
        )
    else:
        grouped_iono_files = None

    # ######################################
    # 1. Burst-wise Wrapped phase estimation
    # ######################################
    if len(grouped_slc_files) > 1:
        logger.info(f"Found SLC files from {len(grouped_slc_files)} bursts")
        wrapped_phase_cfgs = [
            (
                burst,  # Include the burst for logging purposes
                _create_burst_cfg(
                    cfg,
                    burst,
                    grouped_slc_files,
                    grouped_amp_mean_files,
                    grouped_amp_dispersion_files,
                ),
            )
            for burst in grouped_slc_files
        ]
        for _, burst_cfg in wrapped_phase_cfgs:
            burst_cfg.create_dir_tree()
        # Remove the mid-level directories which will be empty due to re-grouping
        _remove_dir_if_empty(cfg.phase_linking._directory)
        _remove_dir_if_empty(cfg.ps_options._directory)

    else:
        # grab the only key (either a burst, or "") and use that
        cfg.create_dir_tree()
        b = list(grouped_slc_files.keys())[0]
        wrapped_phase_cfgs = [(b, cfg)]

    ifg_file_list: list[Path] = []
    temp_coh_file_list: list[Path] = []
    ps_file_list: list[Path] = []
    # The comp_slc tracking object is a dict, since we'll need to organize
    # multiple comp slcs by burst (they'll have the same filename)
    comp_slc_dict: dict[str, Path] = {}
    # Now for each burst, run the wrapped phase estimation
    # Try running several bursts in parallel...
    Executor = (
        ProcessPoolExecutor
        if cfg.worker_settings.n_parallel_bursts > 1
        else DummyProcessPoolExecutor
    )
    mw = cfg.worker_settings.n_parallel_bursts
    ctx = mp.get_context("spawn")
    with Executor(max_workers=mw, mp_context=ctx) as exc:
        fut_to_burst = {
            exc.submit(
                wrapped_phase.run,
                burst_cfg,
                debug=debug,
            ): burst
            for burst, burst_cfg in wrapped_phase_cfgs
        }
        for fut in fut_to_burst:
            burst = fut_to_burst[fut]

            cur_ifg_list, comp_slc, temp_coh, ps_file = fut.result()
            ifg_file_list.extend(cur_ifg_list)
            comp_slc_dict[burst] = comp_slc
            temp_coh_file_list.append(temp_coh)
            ps_file_list.append(ps_file)

    # ###################################
    # 2. Stitch burst-wise interferograms
    # ###################################

    # TODO: figure out how best to pick the corr size
    # Is there one best size? dependent on `half_window` or resolution?
    # For now, just pick a reasonable size
    corr_window_size = (11, 11)
    (
        stitched_ifg_paths,
        stitched_cor_paths,
        stitched_temp_coh_file,
        stitched_ps_file,
    ) = stitching_bursts.run(
        ifg_file_list=ifg_file_list,
        temp_coh_file_list=temp_coh_file_list,
        ps_file_list=ps_file_list,
        stitched_ifg_dir=cfg.interferogram_network._directory,
        output_options=cfg.output_options,
        file_date_fmt=cfg.input_options.cslc_date_fmt,
        corr_window_size=corr_window_size,
    )

    # ###################################
    # 3. Unwrap stitched interferograms
    # ###################################
    if not cfg.unwrap_options.run_unwrap:
        logger.info("Skipping unwrap step")
        _print_summary(cfg)
        return

    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks
    unwrapping.run(
        ifg_file_list=stitched_ifg_paths,
        cor_file_list=stitched_cor_paths,
        nlooks=nlooks,
        unwrap_options=cfg.unwrap_options,
        mask_file=cfg.mask_file,
    )

    # ##############################################
    # 4. Estimate corrections for each interferogram
    # ##############################################

    stitched_ifg_dir = cfg.interferogram_network._directory
    ifg_filenames = sorted(Path(stitched_ifg_dir).glob("*.int"))
    out_dir = cfg.work_directory / cfg.correction_options._atm_directory
    out_dir.mkdir(exist_ok=True)
    grouped_slc_files = group_by_date(cfg.cslc_file_list)

    # Troposphere
    if cfg.correction_options.dem_file is None:
        logger.warning(
            "DEM file is not given, skip estimating tropospheric corrections..."
        )
    else:
        estimate_tropospheric_delay(
            ifg_file_list=ifg_filenames,
            troposphere_files=grouped_tropo_files,
            slc_files=grouped_slc_files,
            geom_files=cfg.correction_options.geometry_files,
            dem_file=cfg.correction_options.dem_file,
            output_dir=out_dir,
            tropo_package=cfg.correction_options.tropo_package,
            tropo_model=cfg.correction_options.tropo_model,
            tropo_delay_type=cfg.correction_options.tropo_delay_type,
            strides=cfg.output_options.strides,
        )

    # Ionosphere
    estimate_ionospheric_delay(
        ifg_file_list=ifg_filenames,
        slc_files=grouped_slc_files,
        tec_files=grouped_iono_files,
        geom_files=cfg.correction_options.geometry_files,
        output_dir=out_dir,
        strides=cfg.output_options.strides,
    )

    # Print the maximum memory usage for each worker
    _print_summary(cfg)


def _print_summary(cfg):
    """Print the maximum memory usage and version info."""
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")
