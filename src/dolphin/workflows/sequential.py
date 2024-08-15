"""Estimate wrapped phase using batches of ministacks.

Initially based on [@Ansari2017SequentialEstimatorEfficient].
"""

from __future__ import annotations

import logging
from itertools import chain
from os import fspath
from pathlib import Path
from typing import Optional

from osgeo_utils import gdal_calc

from dolphin import io
from dolphin._types import Filename
from dolphin.io import VRTStack
from dolphin.stack import MiniStackPlanner

from .config import ShpMethod
from .single import run_wrapped_phase_single

logger = logging.getLogger(__name__)

__all__ = ["run_wrapped_phase_sequential"]


def run_wrapped_phase_sequential(
    *,
    slc_vrt_file: Filename,
    ministack_planner: MiniStackPlanner,
    ministack_size: int,
    manual_reference_idx: int | None = None,
    half_window: dict,
    strides: Optional[dict] = None,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    use_evd: bool = False,
    beta: float = 0.00,
    block_shape: tuple[int, int] = (512, 512),
    baseline_lag: Optional[int] = None,
    **tqdm_kwargs,
) -> tuple[list[Path], list[Path], Path, Path]:
    """Estimate wrapped phase using batches of ministacks."""
    if strides is None:
        strides = {"x": 1, "y": 1}
    output_folder = ministack_planner.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    ministacks = ministack_planner.plan(
        ministack_size,
        manual_idxs=[manual_reference_idx] if manual_reference_idx is not None else [],
    )

    v_all = VRTStack.from_vrt_file(slc_vrt_file)
    logger.info(f"Full file range: {v_all.file_list[0]} to {v_all.file_list[-1]}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Number of ministacks of size {ministack_size}: {len(ministacks)}")

    if shp_nslc is None:
        shp_nslc = v_all.shape[0]

    # list where each item is [output_slc_files] from a ministack
    output_slc_files: list[list] = []
    # Each item is the temp_coh/shp_count file from a ministack
    temp_coh_files: list[Path] = []
    shp_count_files: list[Path] = []

    # function to check if a ministack has already been processed
    def already_processed(d: Path, search_ext: str = ".tif") -> bool:
        return d.exists() and len(list(d.glob(f"*{search_ext}"))) > 0

    # Solve each ministack using the current chunk (and the previous compressed SLCs)
    for ministack in ministacks:
        cur_output_folder = ministack.output_folder

        if already_processed(cur_output_folder):
            logger.info(f"Skipping {cur_output_folder} because it already exists.")
        else:
            cur_files = ministack.file_list
            start_end = ministack.real_slc_date_range_str
            logger.info(
                f"Processing {len(cur_files)} SLCs. Output folder: {cur_output_folder}"
            )
            cur_vrt = VRTStack(
                cur_files,
                outfile=output_folder / f"{start_end}.vrt",
                sort_files=False,
                subdataset=v_all.subdataset,
            )
            # # Run the phase linking process on the current ministack
            # reference_idx = max(0, first_real_slc_idx - 1)

            # Currently: we are always using the first SLC as the reference,
            # even if this is a compressed SLC.
            # Will need to change this if we want to accommodate the original
            # Sequential Estimator+Datum Adjustment method.

            run_wrapped_phase_single(
                slc_vrt_file=cur_vrt,
                ministack=ministack,
                output_folder=cur_output_folder,
                half_window=half_window,
                strides=strides,
                compressed_reference_idx=ministack.reference_idx,
                use_evd=use_evd,
                beta=beta,
                mask_file=mask_file,
                ps_mask_file=ps_mask_file,
                amp_mean_file=amp_mean_file,
                amp_dispersion_file=amp_dispersion_file,
                shp_method=shp_method,
                shp_alpha=shp_alpha,
                shp_nslc=shp_nslc,
                block_shape=block_shape,
                baseline_lag=baseline_lag,
                **tqdm_kwargs,
            )

        cur_output_files, cur_comp_slc_file, temp_coh_file, shp_count_file = (
            _get_outputs_from_folder(cur_output_folder)
        )
        output_slc_files.append(cur_output_files)
        temp_coh_files.append(temp_coh_file)
        shp_count_files.append(shp_count_file)

    ##############################################

    # Average the temporal coherence files in each ministack
    full_span = ministack_planner.real_slc_date_range_str
    output_temp_coh_file = output_folder / f"temporal_coherence_average_{full_span}.tif"
    output_shp_count_file = output_folder / f"shp_counts_average_{full_span}.tif"

    # we can pass the list of files to gdal_calc, which interprets it
    # as a multi-band file
    _average_rasters(temp_coh_files, output_temp_coh_file, "Float32")
    _average_rasters(shp_count_files, output_shp_count_file, "Int16")

    # Combine the separate SLC output lists into a single list
    all_slc_files = list(chain.from_iterable(output_slc_files))
    all_comp_slc_files = [ms.get_compressed_slc_info().path for ms in ministacks]

    out_pl_slcs = []
    for slc_fname in all_slc_files:
        slc_fname.rename(output_folder / slc_fname.name)
        out_pl_slcs.append(output_folder / slc_fname.name)

    comp_slc_outputs = []
    for p in all_comp_slc_files:
        p.rename(output_folder / p.name)
        comp_slc_outputs.append(output_folder / p.name)

    return out_pl_slcs, comp_slc_outputs, output_temp_coh_file, output_shp_count_file


def _get_outputs_from_folder(
    output_folder: Path,
) -> tuple[list[Path], Path, Path, Path]:
    cur_output_files = sorted(output_folder.glob("2*.slc.tif"))
    cur_comp_slc_file = next(output_folder.glob("compressed_*"))
    temp_coh_file = next(output_folder.glob("temporal_coherence_*"))
    shp_count_file = next(output_folder.glob("shp_counts_*"))
    # Currently ignoring to not stitch:
    # eigenvalues, estimator, avg_coh
    return cur_output_files, cur_comp_slc_file, temp_coh_file, shp_count_file


def _average_rasters(file_list: list[Path], outfile: Path, output_type: str):
    if len(file_list) == 1:
        file_list[0].rename(outfile)
        return

    logger.info(f"Averaging {len(file_list)} files into {outfile}")
    gdal_calc.Calc(
        NoDataValue=0,
        format="GTiff",
        outfile=fspath(outfile),
        type=output_type,
        quiet=True,
        overwrite=True,
        creation_options=io.DEFAULT_TIFF_OPTIONS,
        A=file_list,
        calc="numpy.nanmean(A, axis=0)",
    )
