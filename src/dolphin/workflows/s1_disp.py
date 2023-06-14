#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pformat
from typing import Optional

from dolphin import __version__
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin.utils import get_max_memory_usage, set_num_threads

from . import _product, stitch_and_unwrap, wrapped_phase
from ._pge_runconfig import RunConfig
from ._utils import group_by_burst
from .config import Workflow


@log_runtime
def run(
    cfg: Workflow,
    debug: bool = False,
    pge_runconfig: Optional[RunConfig] = None,
):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object for controlling the
        workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    pge_runconfig : RunConfig, optional
        If provided, adds PGE-specific metadata to the output product.
        Not used by the workflow itself, only for extra metadata.
    """
    # Set the logging level for all `dolphin.` modules
    logger = get_log(name="dolphin", debug=debug, filename=cfg.log_file)
    logger.debug(pformat(cfg.dict()))
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
    with Executor(max_workers=cfg.worker_settings.n_parallel_bursts) as exc:
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

            cur_ifg_list, comp_slc, tcorr = fut.result()
            ifg_file_list.extend(cur_ifg_list)
            comp_slc_dict[burst] = comp_slc
            tcorr_file_list.append(tcorr)

    # ###################################
    # 2. Stitch and unwrap interferograms
    # ###################################
    unwrapped_paths, conncomp_paths, spatial_corr_paths, stitched_tcorr_file = (
        stitch_and_unwrap.run(
            ifg_file_list=ifg_file_list,
            tcorr_file_list=tcorr_file_list,
            cfg=cfg,
            debug=debug,
        )
    )

    # ######################################
    # 3. Finalize the output as an HDF5 product
    # ######################################
    logger.info(f"Creating {len(unwrapped_paths)} outputs in {cfg.output_directory}")
    for unw_p, cc_p, s_corr_p in zip(
        unwrapped_paths,
        conncomp_paths,
        spatial_corr_paths,
    ):
        output_name = cfg.output_directory / unw_p.with_suffix(".nc").name
        _product.create_output_product(
            unw_filename=unw_p,
            conncomp_filename=cc_p,
            tcorr_filename=stitched_tcorr_file,
            spatial_corr_filename=s_corr_p,
            # TODO: How am i going to create the output name?
            # output_name=cfg.outputs.output_name,
            output_name=output_name,
            corrections={},
            pge_runconfig=pge_runconfig,
        )

    if cfg.save_compressed_slc:
        logger.info(f"Saving {len(comp_slc_dict.items())} compressed SLCs")
        output_dir = cfg.output_directory / "compressed_slcs"
        output_dir.mkdir(exist_ok=True)
        _product.create_compressed_products(
            comp_slc_dict=comp_slc_dict,
            output_dir=output_dir,
        )

    # Print the maximum memory usage for each worker
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg.dolphin_version}")
    logger.info(f"Current running dolphin version: {__version__}")


def _create_burst_cfg(
    cfg: Workflow,
    burst_id: str,
    grouped_slc_files: dict[str, list[Path]],
    grouped_amp_mean_files: dict[str, list[Path]],
    grouped_amp_dispersion_files: dict[str, list[Path]],
) -> Workflow:
    cfg_temp_dict = cfg.copy(deep=True, exclude={"cslc_file_list"}).dict()

    # Just update the inputs and the scratch directory
    top_level_scratch = cfg_temp_dict["scratch_directory"]
    cfg_temp_dict.update({"scratch_directory": top_level_scratch / burst_id})
    cfg_temp_dict["cslc_file_list"] = grouped_slc_files[burst_id]
    cfg_temp_dict["amplitude_mean_files"] = grouped_amp_mean_files[burst_id]
    cfg_temp_dict["amplitude_dispersion_files"] = grouped_amp_dispersion_files[burst_id]
    return Workflow(**cfg_temp_dict)


def _remove_dir_if_empty(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass
