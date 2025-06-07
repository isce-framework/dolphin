#!/usr/bin/env python
from __future__ import annotations

import logging

# import contextlib
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from opera_utils import group_by_burst
from tqdm.auto import tqdm

from dolphin import __version__, timeseries, utils
from dolphin._log import log_runtime, setup_logging
from dolphin.timeseries import ReferencePoint

from . import stitching_bursts, unwrapping, wrapped_phase
from ._utils import _create_burst_cfg, _remove_dir_if_empty
from .config import DisplacementWorkflow

logger = logging.getLogger("dolphin")


@dataclass
class OutputPaths:
    """Output files of the `DisplacementWorkflow`."""

    comp_slc_dict: dict[str, list[Path]]
    stitched_ifg_paths: list[Path]
    stitched_cor_paths: list[Path]
    stitched_temp_coh_file: Path
    stitched_ps_file: Path
    stitched_amp_dispersion_file: Path
    stitched_shp_count_file: Path
    stitched_similarity_file: Path
    unwrapped_paths: list[Path] | None
    conncomp_paths: list[Path] | None
    timeseries_paths: list[Path] | None
    timeseries_residual_paths: list[Path] | None
    reference_point: ReferencePoint | None


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    debug: bool = False,
) -> OutputPaths:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    if cfg.log_file is None:
        cfg.log_file = cfg.work_directory / "dolphin.log"
    # Set the logging level for all `dolphin.` modules
    setup_logging(logger_name="dolphin", debug=debug, filename=cfg.log_file)
    # TODO: need to pass the cfg filename for the logger
    logger.debug(cfg.model_dump())

    if not cfg.worker_settings.gpu_enabled:
        utils.disable_gpu()
    utils.set_num_threads(cfg.worker_settings.threads_per_worker)

    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise
        # Otherwise, we have SLC files which are not OPERA burst files
        grouped_slc_files = {"phase_linking": cfg.cslc_file_list}

    if cfg.amplitude_dispersion_files:
        grouped_amp_dispersion_files = group_by_burst(cfg.amplitude_dispersion_files)
    else:
        grouped_amp_dispersion_files = defaultdict(list)
    if cfg.amplitude_mean_files:
        grouped_amp_mean_files = group_by_burst(cfg.amplitude_mean_files)
    else:
        grouped_amp_mean_files = defaultdict(list)
    if cfg.layover_shadow_mask_files:
        grouped_layover_shadow_mask_files = group_by_burst(
            cfg.layover_shadow_mask_files
        )
    else:
        grouped_layover_shadow_mask_files = defaultdict(list)

    # ######################################
    # 1. Burst-wise Wrapped phase estimation
    # ######################################
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
                grouped_layover_shadow_mask_files,
            ),
        )
        for burst in grouped_slc_files
    ]
    for _, burst_cfg in wrapped_phase_cfgs:
        burst_cfg.create_dir_tree()
        # Remove the mid-level directories which will be empty due to re-grouping
        _remove_dir_if_empty(burst_cfg.timeseries_options._directory)
        _remove_dir_if_empty(burst_cfg.unwrap_options._directory)

    ifg_file_list: list[Path] = []
    temp_coh_file_list: list[Path] = []
    ps_file_list: list[Path] = []
    amp_dispersion_file_list: list[Path] = []
    shp_count_file_list: list[Path] = []
    similarity_file_list: list[Path] = []
    # The comp_slc tracking object is a dict, since we'll need to organize
    # multiple comp slcs by burst (they'll have the same filename)
    comp_slc_dict: dict[str, list[Path]] = {}
    # Now for each burst, run the wrapped phase estimation
    # Try running several bursts in parallel...
    # Use the Dummy one if not going parallel, as debugging is much simpler
    num_workers = cfg.worker_settings.n_parallel_bursts
    num_parallel = min(num_workers, len(grouped_slc_files))
    Executor = (
        ProcessPoolExecutor if num_parallel > 1 else utils.DummyProcessPoolExecutor
    )
    workers_per_burst = num_workers // num_parallel
    ctx = mp.get_context("spawn")
    tqdm.set_lock(ctx.RLock())
    with Executor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=tqdm.set_lock,
        initargs=(tqdm.get_lock(),),
    ) as exc:
        fut_to_burst = {
            exc.submit(
                wrapped_phase.run,
                burst_cfg,
                debug=debug,
                max_workers=workers_per_burst,
                tqdm_kwargs={
                    "position": i,
                },
            ): burst
            for i, (burst, burst_cfg) in enumerate(wrapped_phase_cfgs)
        }
        for fut, burst in fut_to_burst.items():
            (
                cur_ifg_list,
                comp_slcs,
                temp_coh,
                ps_file,
                amp_disp_file,
                shp_count,
                similarity,
            ) = fut.result()
            ifg_file_list.extend(cur_ifg_list)
            comp_slc_dict[burst] = comp_slcs
            temp_coh_file_list.append(temp_coh)
            ps_file_list.append(ps_file)
            amp_dispersion_file_list.append(amp_disp_file)
            shp_count_file_list.append(shp_count)
            similarity_file_list.append(similarity)

    # ###################################
    # 2. Stitch burst-wise interferograms
    # ###################################

    # TODO: figure out how best to pick the corr size
    # Is there one best size? dependent on `half_window` or resolution?
    # For now, just pick a reasonable size
    corr_window_size = (11, 11)
    stitched_paths = stitching_bursts.run(
        ifg_file_list=ifg_file_list,
        temp_coh_file_list=temp_coh_file_list,
        ps_file_list=ps_file_list,
        amp_dispersion_list=amp_dispersion_file_list,
        shp_count_file_list=shp_count_file_list,
        similarity_file_list=similarity_file_list,
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
        return OutputPaths(
            comp_slc_dict=comp_slc_dict,
            stitched_ifg_paths=stitched_paths.ifg_paths,
            stitched_cor_paths=stitched_paths.interferometric_corr_paths,
            stitched_temp_coh_file=stitched_paths.temp_coh_file,
            stitched_ps_file=stitched_paths.ps_file,
            stitched_amp_dispersion_file=stitched_paths.amp_dispersion_file,
            stitched_shp_count_file=stitched_paths.shp_count_file,
            stitched_similarity_file=stitched_paths.similarity_file,
            unwrapped_paths=None,
            conncomp_paths=None,
            timeseries_paths=None,
            timeseries_residual_paths=None,
            reference_point=None,
        )

    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks
    unwrapped_paths, conncomp_paths = unwrapping.run(
        ifg_file_list=stitched_paths.ifg_paths,
        cor_file_list=stitched_paths.interferometric_corr_paths,
        temporal_coherence_filename=stitched_paths.temp_coh_file,
        similarity_filename=stitched_paths.similarity_file,
        nlooks=nlooks,
        unwrap_options=cfg.unwrap_options,
        mask_file=cfg.mask_file,
    )

    # ##############################################
    # 4. If a network was unwrapped, invert it
    # ##############################################

    ts_opts = cfg.timeseries_options
    # Skip if we didn't ask for inversion/velocity
    if ts_opts.run_inversion or ts_opts.run_velocity:
        # the output of run_timeseries is not currently used so pre-commit removes it
        # let's add back if we need it
        timeseries_paths, timeseries_residual_paths, reference_point = timeseries.run(
            unwrapped_paths=unwrapped_paths,
            conncomp_paths=conncomp_paths,
            corr_paths=stitched_paths.interferometric_corr_paths,
            # TODO: Right now we don't have the option to pick a different candidate
            # or quality file. Figure out if this is worth exposing
            reference_point=cfg.timeseries_options.reference_point,
            quality_file=stitched_paths.temp_coh_file,
            reference_candidate_threshold=0.95,
            output_dir=ts_opts._directory,
            method=timeseries.InversionMethod(ts_opts.method),
            run_velocity=ts_opts.run_velocity,
            velocity_file=ts_opts._velocity_file,
            correlation_threshold=ts_opts.correlation_threshold,
            num_threads=ts_opts.num_parallel_blocks,
            # TODO: do i care to configure block shape, or num threads from somewhere?
            # num_threads=cfg.worker_settings....?
            wavelength=cfg.input_options.wavelength,
            add_overviews=cfg.output_options.add_overviews,
            extra_reference_date=cfg.output_options.extra_reference_date,
        )

    else:
        timeseries_paths = None
        timeseries_residual_paths = None
        reference_point = None

    # Print the maximum memory usage for each worker
    _print_summary(cfg)
    return OutputPaths(
        comp_slc_dict=comp_slc_dict,
        stitched_ifg_paths=stitched_paths.ifg_paths,
        stitched_cor_paths=stitched_paths.interferometric_corr_paths,
        stitched_temp_coh_file=stitched_paths.temp_coh_file,
        stitched_ps_file=stitched_paths.ps_file,
        stitched_amp_dispersion_file=stitched_paths.amp_dispersion_file,
        stitched_shp_count_file=stitched_paths.shp_count_file,
        stitched_similarity_file=stitched_paths.similarity_file,
        unwrapped_paths=unwrapped_paths,
        # TODO: Let's keep the unwrapped_paths since all the outputs are
        # corresponding to those and if we have a network unwrapping, the
        # inversion would create different single-reference network and we need
        # to update other products like conncomp
        # unwrapped_paths=inverted_phase_paths,
        conncomp_paths=conncomp_paths,
        timeseries_paths=timeseries_paths,
        timeseries_residual_paths=timeseries_residual_paths,
        reference_point=reference_point,
    )


def _print_summary(cfg):
    """Print the maximum memory usage and version info."""
    max_mem = utils.get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")
