#!/usr/bin/env python
from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pformat

from dolphin import __version__
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin.opera_utils import group_by_burst
from dolphin.utils import get_max_memory_usage, set_num_threads

from . import stitch_and_unwrap, wrapped_phase
from ._utils import _create_burst_cfg, _remove_dir_if_empty
from .config import Workflow


@log_runtime
def run(
    cfg: Workflow,
    debug: bool = False,
):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object for controlling the
        workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    """
    # Set the logging level for all `dolphin.` modules
    logger = get_log(name="dolphin", debug=debug, filename=cfg.log_file)
    logger.debug(pformat(cfg.model_dump()))
    cfg.create_dir_tree(debug=debug)

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
        grouped_amp_dispersion_files = group_by_burst(
            cfg.amplitude_dispersion_files, minimum_images=1
        )
    else:
        grouped_amp_dispersion_files = defaultdict(list)
    if cfg.amplitude_mean_files:
        grouped_amp_mean_files = group_by_burst(
            cfg.amplitude_mean_files, minimum_images=1
        )
    else:
        grouped_amp_mean_files = defaultdict(list)

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
        b = list(grouped_slc_files.keys())[0]
        wrapped_phase_cfgs = [(b, cfg)]

    ifg_file_list: list[Path] = []
    tcorr_file_list: list[Path] = []
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

            cur_ifg_list, comp_slc, tcorr, ps_file = fut.result()
            ifg_file_list.extend(cur_ifg_list)
            comp_slc_dict[burst] = comp_slc
            tcorr_file_list.append(tcorr)
            ps_file_list.append(ps_file)

    # ###################################
    # 2. Stitch and unwrap interferograms
    # ###################################
    unwrapped_paths, conncomp_paths, spatial_corr_paths, stitched_tcorr_file = (
        stitch_and_unwrap.run(
            ifg_file_list=ifg_file_list,
            tcorr_file_list=tcorr_file_list,
            ps_file_list=ps_file_list,
            cfg=cfg,
            debug=debug,
            unwrap_jobs=cfg.unwrap_options.n_parallel_jobs,
        )
    )

    # Print the maximum memory usage for each worker
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")
