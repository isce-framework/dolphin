#!/usr/bin/env python
from __future__ import annotations

import logging

# import contextlib
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opera_utils import group_by_burst
from tqdm.auto import tqdm

from dolphin import __version__, timeseries, utils
from dolphin._log import log_runtime, setup_logging
from dolphin._types import Bbox
from dolphin.timeseries import ReferencePoint

from . import stitching_bursts, unwrapping, wrapped_phase
from ._block_split import BlockBounds, crop_to_central, split_frame_into_blocks
from ._utils import _create_burst_cfg, _remove_dir_if_empty
from .config import DisplacementWorkflow

logger = logging.getLogger("dolphin")


def _worker_init(tqdm_lock: Any, disable_hdf5_locking: bool) -> None:
    """Initialize a wrapped-phase ProcessPoolExecutor worker process.

    Sets the shared tqdm lock and, when ``disable_hdf5_locking`` is True,
    disables HDF5 file locking in this child process. The HDF5 env var only
    matters for the worker that opens the files, so setting it here keeps
    the parent process's env clean (libraries imported in the parent that
    cached the env value at import time won't be surprised by a mutated
    parent env).
    """
    tqdm.set_lock(tqdm_lock)
    if disable_hdf5_locking:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


@dataclass
class _GroupedInputs:
    """Per-burst input file groupings + per-block bounds.

    Built once at the top of :func:`run` by :func:`_prepare_grouped_inputs`.
    Each ``grouped_*`` dict maps a burst id — either a real OPERA burst id
    (``"t087_185678_iw2"``) or a synthetic NISAR block id
    (``"block_00"`` / ``"phase_linking"``) — to that unit's file list.

    ``block_bounds`` is non-empty only when NISAR-style inputs were split
    into multiple azimuth blocks via ``cfg.input_options.azimuth_blocks``.
    """

    is_opera_burst_mode: bool
    grouped_slc_files: dict[str, list[Path]]
    block_bounds: dict[str, BlockBounds]
    grouped_amp_dispersion_files: dict[str, list[Path]]
    grouped_amp_mean_files: dict[str, list[Path]]
    grouped_layover_shadow_mask_files: dict[str, list[Path]]


def _prepare_grouped_inputs(cfg: DisplacementWorkflow) -> _GroupedInputs:
    """Detect input mode and group all file lists into per-burst dicts.

    Tries OPERA's ``group_by_burst`` first. If the CSLC filenames don't carry
    OPERA burst ids (e.g. NISAR GSLCs), falls back to one of two non-OPERA
    paths:

    1. ``azimuth_blocks > 1`` — splits the single frame into N synthetic
       blocks via :func:`split_frame_into_blocks`. Each block reuses the full
       CSLC list and gets per-block ``output_options.bounds`` (returned in
       ``block_bounds``).
    2. ``azimuth_blocks == 1`` — keeps the full frame as a single "burst"
       under the key ``"phase_linking"``, no bounds.

    Optional file lists (amp_disp, amp_mean, layover_shadow) are grouped by
    burst in OPERA mode or broadcast to every key in non-OPERA mode. The
    bounds mask inside ``wrapped_phase._get_mask`` clips them per block
    automatically when bounds are set.
    """
    is_opera_burst_mode = True
    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
        block_bounds: dict[str, BlockBounds] = {}
    except ValueError as e:
        if "Could not parse burst id" not in str(e):
            raise
        # Non-OPERA-burst inputs (e.g. NISAR full-frame GSLCs).
        is_opera_burst_mode = False
        if cfg.input_options.azimuth_blocks > 1:
            # Split the single frame into N synthetic "bursts" for parallelism.
            block_bounds = split_frame_into_blocks(
                cfg,
                num_blocks=cfg.input_options.azimuth_blocks,
                halo_rows=cfg.input_options.halo_rows,
            )
            grouped_slc_files = dict.fromkeys(block_bounds, cfg.cslc_file_list)
        else:
            # Full-frame run: a single "burst" containing all inputs.
            grouped_slc_files = {"phase_linking": cfg.cslc_file_list}
            block_bounds = {}

    def _group_or_broadcast(
        files: list[Path] | None,
    ) -> dict[str, list[Path]]:
        if not files:
            return defaultdict(list)
        if is_opera_burst_mode:
            return group_by_burst(files)
        return {key: list(files) for key in grouped_slc_files}

    return _GroupedInputs(
        is_opera_burst_mode=is_opera_burst_mode,
        grouped_slc_files=grouped_slc_files,
        block_bounds=block_bounds,
        grouped_amp_dispersion_files=_group_or_broadcast(
            cfg.amplitude_dispersion_files
        ),
        grouped_amp_mean_files=_group_or_broadcast(cfg.amplitude_mean_files),
        grouped_layover_shadow_mask_files=_group_or_broadcast(
            cfg.layover_shadow_mask_files
        ),
    )


@dataclass
class _GroupedInputs:
    """Per-burst input file groupings + per-block bounds.

    Built once at the top of :func:`run` by :func:`_prepare_grouped_inputs`.
    Each ``grouped_*`` dict maps a burst id — either a real OPERA burst id
    (``"t087_185678_iw2"``) or a synthetic NISAR block id
    (``"block_00"`` / ``"phase_linking"``) — to that unit's file list.

    ``block_bounds`` is non-empty only when NISAR-style inputs were split
    into multiple azimuth blocks via ``cfg.input_options.azimuth_blocks``.
    """

    is_opera_burst_mode: bool
    grouped_slc_files: dict[str, list[Path]]
    block_bounds: dict[str, tuple[Bbox, int]]
    grouped_amp_dispersion_files: dict[str, list[Path]]
    grouped_amp_mean_files: dict[str, list[Path]]
    grouped_layover_shadow_mask_files: dict[str, list[Path]]


def _prepare_grouped_inputs(cfg: DisplacementWorkflow) -> _GroupedInputs:
    """Detect input mode and group all file lists into per-burst dicts.

    Tries OPERA's ``group_by_burst`` first. If the CSLC filenames don't carry
    OPERA burst ids (e.g. NISAR GSLCs), falls back to one of two non-OPERA
    paths:

    1. ``azimuth_blocks > 1`` — splits the single frame into N synthetic
       blocks via :func:`split_frame_into_blocks`. Each block reuses the full
       CSLC list and gets per-block ``output_options.bounds`` (returned in
       ``block_bounds``).
    2. ``azimuth_blocks == 1`` — keeps the full frame as a single "burst"
       under the key ``"phase_linking"``, no bounds.

    Optional file lists (amp_disp, amp_mean, layover_shadow) are grouped by
    burst in OPERA mode or broadcast to every key in non-OPERA mode. The
    bounds mask inside ``wrapped_phase._get_mask`` clips them per block
    automatically when bounds are set.
    """
    is_opera_burst_mode = True
    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
        block_bounds: dict[str, tuple[Bbox, int]] = {}
    except ValueError as e:
        if "Could not parse burst id" not in str(e):
            raise
        # Non-OPERA-burst inputs (e.g. NISAR full-frame GSLCs).
        is_opera_burst_mode = False
        if cfg.input_options.azimuth_blocks > 1:
            # Split the single frame into N synthetic "bursts" for parallelism.
            block_bounds = split_frame_into_blocks(
                cfg,
                num_blocks=cfg.input_options.azimuth_blocks,
                halo_rows=cfg.input_options.halo_rows,
            )
            grouped_slc_files = dict.fromkeys(block_bounds, cfg.cslc_file_list)
        else:
            # Full-frame run: a single "burst" containing all inputs.
            grouped_slc_files = {"phase_linking": cfg.cslc_file_list}
            block_bounds = {}

    def _group_or_broadcast(
        files: list[Path] | None,
    ) -> dict[str, list[Path]]:
        if not files:
            return defaultdict(list)
        if is_opera_burst_mode:
            return group_by_burst(files)
        return {key: list(files) for key in grouped_slc_files}

    return _GroupedInputs(
        is_opera_burst_mode=is_opera_burst_mode,
        grouped_slc_files=grouped_slc_files,
        block_bounds=block_bounds,
        grouped_amp_dispersion_files=_group_or_broadcast(
            cfg.amplitude_dispersion_files
        ),
        grouped_amp_mean_files=_group_or_broadcast(cfg.amplitude_mean_files),
        grouped_layover_shadow_mask_files=_group_or_broadcast(
            cfg.layover_shadow_mask_files
        ),
    )


@dataclass
class OutputPaths:
    """Output files of the `DisplacementWorkflow`."""

    comp_slc_dict: dict[str, list[Path]]
    stitched_ifg_paths: list[Path]
    stitched_cor_paths: list[Path]
    stitched_temp_coh_files: list[Path]
    stitched_shp_count_files: list[Path]
    stitched_similarity_files: list[Path]
    stitched_crlb_files: list[Path]
    stitched_closure_phase_files: list[Path]
    stitched_ps_file: Path
    stitched_amp_dispersion_file: Path
    unwrapped_paths: list[Path] | None
    conncomp_paths: list[Path] | None
    timeseries_paths: list[Path] | None
    timeseries_residual_paths: list[Path] | None
    reference_point: ReferencePoint | None

    @property
    def stitched_temp_coh_file(self) -> Path:
        """Backward compatibility: return the last temporal coherence file."""
        return self.stitched_temp_coh_files[-1]

    @property
    def stitched_shp_count_file(self) -> Path:
        """Backward compatibility: return the last SHP count file."""
        return self.stitched_shp_count_files[-1]

    @property
    def stitched_similarity_file(self) -> Path:
        """Backward compatibility: return the last similarity file."""
        return self.stitched_similarity_files[-1]


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    debug: bool = False,
    raise_on_empty: bool = True,
) -> OutputPaths:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    raise_on_empty : bool
        If True, raises a `MaskingError` on the creation of a mask file with
        no valid pixels.
        Otherwise, raises a warning.
        Default is True.

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

    grouped = _prepare_grouped_inputs(cfg)
    num_parallel = min(
        cfg.worker_settings.n_parallel_bursts, len(grouped.grouped_slc_files)
    )
    grouped_slc_files = grouped.grouped_slc_files
    block_bounds = grouped.block_bounds
    grouped_amp_dispersion_files = grouped.grouped_amp_dispersion_files
    grouped_amp_mean_files = grouped.grouped_amp_mean_files
    grouped_layover_shadow_mask_files = grouped.grouped_layover_shadow_mask_files

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
                bounds=block_bounds.get(burst),
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
    crlb_files: list[Path] = []
    closure_phase_files: list[Path] = []
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
    Executor = (
        ProcessPoolExecutor if num_parallel > 1 else utils.DummyProcessPoolExecutor
    )
    workers_per_burst = cfg.worker_settings.n_parallel_bursts // num_parallel
    ctx = mp.get_context("spawn")
    tqdm.set_lock(ctx.RLock())
    # When azimuth-block-as-burst is active and we have >1 parallel worker,
    # multiple processes read the same NISAR HDF5 files at different spatial
    # extents. Disable HDF5 file locking in each worker (not the parent) so
    # concurrent readers don't block on NFS/Lustre flock.
    needs_hdf5_unlock = not grouped.is_opera_burst_mode and num_parallel > 1
    with Executor(
        max_workers=cfg.worker_settings.n_parallel_bursts,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(tqdm.get_lock(), needs_hdf5_unlock),
    ) as exc:
        fut_to_burst = {
            exc.submit(
                wrapped_phase.run,
                burst_cfg,
                debug=debug,
                max_workers=workers_per_burst,
                raise_on_empty=raise_on_empty,
                tqdm_kwargs={
                    "position": i,
                },
            ): burst
            for i, (burst, burst_cfg) in enumerate(wrapped_phase_cfgs)
        }
        for fut, burst in fut_to_burst.items():
            wrapped_phase_output = fut.result()
            (
                cur_ifg_list,
                cur_crlb_files,
                cur_closure_phase_files,
                comp_slcs,
                temp_coh_files,
                ps_file,
                amp_disp_file,
                shp_count_files,
                similarity_files,
            ) = wrapped_phase_output

            # If this burst is a synthetic azimuth block, the per-block outputs
            # cover central rows + a halo. Crop the stitching-bound rasters
            # down to central_bounds so adjacent blocks have disjoint extents
            # and the stitcher doesn't have to pick a winner in the halo
            # overlap. comp_slcs stay full-extent — they feed back into this
            # block's next ministack and need to match its read bounds.
            # crop_to_central may change a path's extension (.vrt -> .tif),
            # so substitute the returned paths back in.
            bb = block_bounds.get(burst)
            if bb is not None and bb.read_bounds != bb.central_bounds:
                cb = bb.central_bounds
                cur_ifg_list = [crop_to_central(f, cb) for f in cur_ifg_list]
                cur_crlb_files = [crop_to_central(f, cb) for f in cur_crlb_files]
                cur_closure_phase_files = [
                    crop_to_central(f, cb) for f in cur_closure_phase_files
                ]
                temp_coh_files = [crop_to_central(f, cb) for f in temp_coh_files]
                shp_count_files = [crop_to_central(f, cb) for f in shp_count_files]
                similarity_files = [crop_to_central(f, cb) for f in similarity_files]
                ps_file = crop_to_central(ps_file, cb)
                amp_disp_file = crop_to_central(amp_disp_file, cb)

            ifg_file_list.extend(cur_ifg_list)
            crlb_files.extend(cur_crlb_files)
            closure_phase_files.extend(cur_closure_phase_files)
            comp_slc_dict[burst] = comp_slcs
            temp_coh_file_list.extend(temp_coh_files)
            ps_file_list.append(ps_file)
            amp_dispersion_file_list.append(amp_disp_file)
            shp_count_file_list.extend(shp_count_files)
            similarity_file_list.extend(similarity_files)

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
        crlb_file_list=crlb_files,
        amp_dispersion_list=amp_dispersion_file_list,
        shp_count_file_list=shp_count_file_list,
        similarity_file_list=similarity_file_list,
        closure_phase_file_list=closure_phase_files,
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
            stitched_temp_coh_files=stitched_paths.temp_coh_files,
            stitched_ps_file=stitched_paths.ps_file,
            stitched_amp_dispersion_file=stitched_paths.amp_dispersion_file,
            stitched_shp_count_files=stitched_paths.shp_count_files,
            stitched_similarity_files=stitched_paths.similarity_files,
            stitched_crlb_files=stitched_paths.crlb_paths,
            stitched_closure_phase_files=stitched_paths.closure_phase_files,
            unwrapped_paths=None,
            conncomp_paths=None,
            timeseries_paths=None,
            timeseries_residual_paths=None,
            reference_point=None,
        )

    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks

    # TODO: Not sure if i'll ever want more than one quality file
    # Dividing per-ministack in here would probably be complicated.
    # It'll probably be better to make an entirely different workflow
    # for that.
    avg_temp_coh_file = stitched_paths.temp_coh_files[-1]
    full_similarity_file = stitched_paths.similarity_files[-1]
    unwrapped_paths, conncomp_paths = unwrapping.run(
        ifg_file_list=stitched_paths.ifg_paths,
        cor_file_list=stitched_paths.interferometric_corr_paths,
        temporal_coherence_filename=avg_temp_coh_file,
        similarity_filename=full_similarity_file,
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
            quality_file=avg_temp_coh_file,
            reference_candidate_threshold=0.95,
            output_dir=ts_opts._directory,
            method=timeseries.InversionMethod(ts_opts.method),
            run_velocity=ts_opts.run_velocity,
            velocity_file=ts_opts._velocity_file,
            mask_path=cfg.mask_file if ts_opts.apply_mask_to_timeseries else None,
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
        stitched_temp_coh_files=stitched_paths.temp_coh_files,
        stitched_crlb_files=stitched_paths.crlb_paths,
        stitched_closure_phase_files=stitched_paths.closure_phase_files,
        stitched_ps_file=stitched_paths.ps_file,
        stitched_amp_dispersion_file=stitched_paths.amp_dispersion_file,
        stitched_shp_count_files=stitched_paths.shp_count_files,
        stitched_similarity_files=stitched_paths.similarity_files,
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
