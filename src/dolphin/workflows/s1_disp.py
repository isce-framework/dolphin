#!/usr/bin/env python
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional

from dolphin._log import get_log, log_runtime

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
    logger = get_log(debug=debug, filename=cfg.log_file)
    logger.debug(pformat(cfg.dict()))
    cfg.create_dir_tree(debug=debug)

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
            cfg.amplitude_dispersion_files, minimum_slcs=1
        )
    else:
        grouped_amp_dispersion_files = defaultdict(list)
    if cfg.amplitude_mean_files:
        grouped_amp_mean_files = group_by_burst(
            cfg.amplitude_mean_files, minimum_slcs=1
        )
    else:
        grouped_amp_mean_files = defaultdict(list)

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
        wrapped_phase_cfgs = [("", cfg)]
    # ###########################
    # 1. Wrapped phase estimation
    # ###########################
    ifg_file_list: List[Path] = []
    tcorr_file_list: List[Path] = []
    # The comp_slc tracking object is a dict, since we'll need to organize
    # multiple comp slcs by burst (they'll have the same filename)
    comp_slc_dict: Dict[str, Path] = {}
    # Now for each burst, run the wrapped phase estimation
    for burst, burst_cfg in wrapped_phase_cfgs:
        msg = "Running wrapped phase estimation"
        if burst:
            msg += f" for burst {burst}"
        logger.info(msg)
        logger.debug(pformat(burst_cfg.dict()))
        cur_ifg_list, comp_slc, tcorr = wrapped_phase.run(burst_cfg, debug=debug)

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


def _create_burst_cfg(
    cfg: Workflow,
    burst_id: str,
    grouped_slc_files: Dict[str, List[Path]],
    grouped_amp_mean_files: Dict[str, List[Path]],
    grouped_amp_dispersion_files: Dict[str, List[Path]],
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
