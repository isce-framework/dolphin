from __future__ import annotations

from pathlib import Path
from typing import Optional

from dolphin import ps, stack, utils
from dolphin._background import NvidiaMemoryWatcher
from dolphin._log import get_log, log_runtime
from dolphin.interferogram import Network

from . import _utils, sequential, single
from .config import Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False) -> tuple[list[Path], Path, Path, Path]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.

    Returns
    -------
    list[Path]
        list of Paths to virtual interferograms created.
    Path
        Path the final compressed SLC file created.
    Path
        Path to temporal correlation file created.
        In the case of a single phase linking step, this is the one tcorr file.
        In the case of sequential phase linking, this is the average tcorr file.
    """
    logger = get_log(debug=debug)
    logger.info(f"Running wrapped phase estimation in {cfg.scratch_directory}")

    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        raise ValueError("No input files found")

    # #############################################
    # Make a VRT pointing to the input SLC files
    # #############################################
    subdataset = cfg.input_options.subdataset
    vrt_stack = stack.VRTStack(
        input_file_list,
        subdataset=subdataset,
        outfile=cfg.scratch_directory / "slc_stack.vrt",
    )

    # Make the nodata mask from the polygons, if we're using OPERA CSLCs
    try:
        nodata_mask_file = cfg.scratch_directory / "nodata_mask.tif"
        _utils.make_nodata_mask(
            vrt_stack.file_list, out_file=nodata_mask_file, buffer_pixels=200
        )
    except Exception as e:
        logger.warning(f"Could not make nodata mask: {e}")
        nodata_mask_file = None

    # ###############
    # PS selection
    # ###############
    ps_output = cfg.ps_options._output_file
    if ps_output.exists():
        logger.info(f"Skipping making existing PS file {ps_output}")
    else:
        logger.info(f"Creating persistent scatterer file {ps_output}")
        try:
            existing_amp: Optional[Path] = cfg.amplitude_mean_files[0]
            existing_disp: Optional[Path] = cfg.amplitude_dispersion_files[0]
        except IndexError:
            existing_amp = existing_disp = None

        ps.create_ps(
            slc_vrt_file=vrt_stack.outfile,
            output_file=ps_output,
            output_amp_mean_file=cfg.ps_options._amp_mean_file,
            output_amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
            amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
            existing_amp_dispersion_file=existing_disp,
            existing_amp_mean_file=existing_amp,
            block_shape=cfg.worker_settings.block_shape,
        )

    # Save a looked version of the PS mask too
    strides = cfg.output_options.strides
    ps_looked_file = ps.multilook_ps_mask(
        strides=strides, ps_mask_file=cfg.ps_options._output_file
    )

    # #########################
    # phase linking/EVD step
    # #########################
    pl_path = cfg.phase_linking._directory

    phase_linked_slcs = list(pl_path.glob("2*.tif"))
    if len(phase_linked_slcs) > 0:
        logger.info(f"Skipping EVD step, {len(phase_linked_slcs)} files already exist")
        comp_slc_file = next(pl_path.glob("compressed*tif"))
        tcorr_file = next(pl_path.glob("tcorr*tif"))
    else:
        logger.info(f"Running sequential EMI step in {pl_path}")
        if utils.gpu_is_available():  # Track the GPU mem usage if we're using it
            watcher = NvidiaMemoryWatcher(log_file=pl_path / "nvidia_memory.log")
        else:
            watcher = None

        # TODO: Need a good way to store the nslc attribute in the PS file...
        # If we pre-compute it from some big stack, we need to use that for SHP
        # finding, not use the size of `slc_vrt_file`
        shp_nslc = None
        if cfg.workflow_name == "single":
            phase_linked_slcs, comp_slc_file, tcorr_file = (
                single.run_wrapped_phase_single(
                    slc_vrt_file=vrt_stack.outfile,
                    output_folder=pl_path,
                    half_window=cfg.phase_linking.half_window.dict(),
                    strides=strides,
                    reference_idx=0,
                    beta=cfg.phase_linking.beta,
                    mask_file=nodata_mask_file,
                    ps_mask_file=ps_output,
                    amp_mean_file=cfg.ps_options._amp_mean_file,
                    amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
                    shp_method=cfg.phase_linking.shp_method,
                    shp_alpha=cfg.phase_linking.shp_alpha,
                    shp_nslc=shp_nslc,
                    block_shape=cfg.worker_settings.block_shape,
                    n_workers=cfg.worker_settings.n_workers,
                    gpu_enabled=cfg.worker_settings.gpu_enabled,
                )
            )
        else:
            phase_linked_slcs, comp_slcs, tcorr_file = (
                sequential.run_wrapped_phase_sequential(
                    slc_vrt_file=vrt_stack.outfile,
                    output_folder=pl_path,
                    half_window=cfg.phase_linking.half_window.dict(),
                    strides=strides,
                    beta=cfg.phase_linking.beta,
                    ministack_size=cfg.phase_linking.ministack_size,
                    mask_file=nodata_mask_file,
                    ps_mask_file=ps_output,
                    amp_mean_file=cfg.ps_options._amp_mean_file,
                    amp_dispersion_file=cfg.ps_options._amp_dispersion_file,
                    shp_method=cfg.phase_linking.shp_method,
                    shp_alpha=cfg.phase_linking.shp_alpha,
                    shp_nslc=shp_nslc,
                    block_shape=cfg.worker_settings.block_shape,
                    n_workers=cfg.worker_settings.n_workers,
                    gpu_enabled=cfg.worker_settings.gpu_enabled,
                )
            )
            comp_slc_file = comp_slcs[-1]

        if watcher:
            watcher.notify_finished()

    # ###################################################
    # Form interferograms from estimated wrapped phase
    # ###################################################
    ifg_dir = cfg.interferogram_network._directory
    existing_ifgs = list(ifg_dir.glob("*.int.*"))
    if len(existing_ifgs) > 0:
        logger.info(f"Skipping interferogram step, {len(existing_ifgs)} exists")
        ifg_file_list = existing_ifgs
    else:
        logger.info(
            f"Creating virtual interferograms from {len(phase_linked_slcs)} files"
        )
        network = Network(
            slc_list=phase_linked_slcs,
            reference_idx=cfg.interferogram_network.reference_idx,
            max_bandwidth=cfg.interferogram_network.max_bandwidth,
            max_temporal_baseline=cfg.interferogram_network.max_temporal_baseline,
            indexes=cfg.interferogram_network.indexes,
            outdir=ifg_dir,
        )
        if len(network.ifg_list) == 0:
            raise ValueError("No interferograms were created")
        else:
            ifg_file_list = [ifg.path for ifg in network.ifg_list]  # type: ignore

    return ifg_file_list, comp_slc_file, tcorr_file, ps_looked_file
