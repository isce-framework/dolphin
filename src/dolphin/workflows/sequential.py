"""Estimate wrapped phase using batches of ministacks.

Initially based on [@Ansari2017SequentialEstimatorEfficient].
"""

from __future__ import annotations

import datetime
import logging
from os import fspath
from pathlib import Path
from typing import Optional, Sequence

from opera_utils import get_dates
from osgeo_utils import gdal_calc

from dolphin import io
from dolphin._types import Filename
from dolphin.io import VRTStack
from dolphin.similarity import create_similarities
from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

from .config import ShpMethod
from .single import run_wrapped_phase_single

logger = logging.getLogger("dolphin")

__all__ = ["run_wrapped_phase_sequential"]


def run_wrapped_phase_sequential(
    *,
    slc_vrt_stack: VRTStack,
    output_folder: Path,
    ministack_size: int,
    half_window: dict,
    strides: Optional[dict[str, int]] = None,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    use_evd: bool = False,
    beta: float = 0.00,
    zero_correlation_threshold: float = 0.0,
    similarity_nearest_n: int | None = None,
    compressed_slc_plan: CompressedSlcPlan = CompressedSlcPlan.ALWAYS_FIRST,
    max_num_compressed: int = 100,
    output_reference_idx: int | None = None,
    new_compressed_reference_idx: int | None = None,
    cslc_date_fmt: str = "%Y%m%d",
    write_closure_phase: bool = True,
    write_crlb: bool = True,
    block_shape: tuple[int, int] = (512, 512),
    baseline_lag: Optional[int] = None,
    max_workers: int = 1,
    **tqdm_kwargs,
) -> tuple[
    list[Path], list[Path], list[Path], list[Path], list[Path], list[Path], list[Path]
]:
    """Estimate wrapped phase using batches of ministacks."""
    if strides is None:
        strides = {"x": 1, "y": 1}
    output_folder.mkdir(parents=True, exist_ok=True)
    input_file_list = slc_vrt_stack.file_list

    is_compressed = ["compressed" in str(f).lower() for f in slc_vrt_stack.file_list]
    input_dates = _get_input_dates(input_file_list, is_compressed, cslc_date_fmt)

    ministack_planner = MiniStackPlanner(
        file_list=slc_vrt_stack.file_list,
        dates=input_dates,
        is_compressed=is_compressed,
        output_folder=output_folder,
        max_num_compressed=max_num_compressed,
        output_reference_idx=output_reference_idx,
        compressed_slc_plan=compressed_slc_plan,
    )
    ministacks = ministack_planner.plan(
        ministack_size, compressed_idx=new_compressed_reference_idx
    )

    logger.info(f"File range start: {Path(slc_vrt_stack.file_list[0]).name}")
    logger.info(f"File range end: {Path(slc_vrt_stack.file_list[-1]).name}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Number of ministacks of size {ministack_size}: {len(ministacks)}")

    if shp_nslc is None:
        shp_nslc = slc_vrt_stack.shape[0]

    # list where each item is extended with output_slc_files from a ministack
    output_slc_files: list[Path] = []
    crlb_files: list[Path] = []
    closure_phase_files: list[Path] = []
    temp_coh_files: list[Path] = []
    similarity_files: list[Path] = []
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
                subdataset=slc_vrt_stack.subdataset,
            )

            run_wrapped_phase_single(
                vrt_stack=cur_vrt,
                ministack=ministack,
                output_folder=cur_output_folder,
                half_window=half_window,
                strides=strides,
                use_evd=use_evd,
                beta=beta,
                zero_correlation_threshold=zero_correlation_threshold,
                mask_file=mask_file,
                ps_mask_file=ps_mask_file,
                amp_mean_file=amp_mean_file,
                amp_dispersion_file=amp_dispersion_file,
                shp_method=shp_method,
                shp_alpha=shp_alpha,
                shp_nslc=shp_nslc,
                similarity_nearest_n=similarity_nearest_n,
                write_closure_phase=write_closure_phase,
                write_crlb=write_crlb,
                block_shape=block_shape,
                baseline_lag=baseline_lag,
                max_workers=max_workers,
                **tqdm_kwargs,
            )

        (
            cur_output_files,
            cur_crlb_files,
            cur_closure_phase_files,
            _cur_comp_slc_file,
            temp_coh_file,
            similarity_file,
            shp_count_file,
        ) = _get_outputs_from_folder(cur_output_folder)
        crlb_files.extend(cur_crlb_files)
        closure_phase_files.extend(cur_closure_phase_files)
        output_slc_files.extend(cur_output_files)
        temp_coh_files.append(temp_coh_file)
        similarity_files.append(similarity_file)
        shp_count_files.append(shp_count_file)

    ##############################################
    # Move the per-ministack files into the `output_folder`
    def move_to_output(files: list[Path], output_folder: Path) -> list[Path]:
        tmp: list[Path] = []
        for p in files:
            tmp.append(p.rename(output_folder / p.name))
        return tmp

    temp_coh_files = move_to_output(temp_coh_files, output_folder)
    shp_count_files = move_to_output(shp_count_files, output_folder)
    similarity_files = move_to_output(similarity_files, output_folder)

    # Average the temporal coherence files in each ministack
    full_span = ministack_planner.real_slc_date_range_str
    avg_temp_coh_file = output_folder / f"temporal_coherence_average_{full_span}.tif"
    avg_shp_count_file = output_folder / f"shp_counts_average_{full_span}.tif"

    # we can pass the list of files to gdal_calc, which interprets it
    # as a multi-band file

    if len(temp_coh_files) > 1:
        _average_or_rename(temp_coh_files, avg_temp_coh_file, "Float32")
        _average_or_rename(shp_count_files, avg_shp_count_file, "Int16")
        temp_coh_files.append(avg_temp_coh_file)
        shp_count_files.append(avg_shp_count_file)

    if len(similarity_files) > 1:
        # Create one phase similarity raster on the whole wrapped time series
        full_similarity_file = output_folder / f"similarity_full_{full_span}.tif"
        create_similarities(
            ifg_file_list=cur_output_files,
            output_file=full_similarity_file,
            # TODO: any of these configurable?
            search_radius=11,
            sim_type="median",
            block_shape=block_shape,
            nearest_n=similarity_nearest_n,
            num_threads=2,
            add_overviews=False,
        )
        similarity_files.append(full_similarity_file)

    all_comp_slc_files = [ms.get_compressed_slc_info().path for ms in ministacks]

    out_pl_slcs = []
    for slc_fname in output_slc_files:
        slc_fname.rename(output_folder / slc_fname.name)
        out_pl_slcs.append(output_folder / slc_fname.name)

    comp_slc_outputs = []
    for p in all_comp_slc_files:
        p.rename(output_folder / p.name)
        comp_slc_outputs.append(output_folder / p.name)

    return (
        out_pl_slcs,
        crlb_files,
        closure_phase_files,
        comp_slc_outputs,
        temp_coh_files,
        shp_count_files,
        similarity_files,
    )


def _get_outputs_from_folder(
    output_folder: Path,
) -> tuple[list[Path], list[Path], list[Path], Path, Path, Path, Path]:
    cur_output_files = sorted(output_folder.glob("2*.slc.tif"))

    cur_comp_slc_file = next(output_folder.glob("compressed_*"))
    temp_coh_file = next(output_folder.glob("temporal_coherence_*"))
    similarity_file = next(output_folder.glob("similarity*"))
    shp_count_file = next(output_folder.glob("shp_counts_*"))
    crlb_files = sorted(output_folder.glob("crlb/crlb*tif"))
    closure_phase_files = sorted(output_folder.glob("closure_phases/closure_phase*tif"))

    return (
        cur_output_files,
        crlb_files,
        closure_phase_files,
        cur_comp_slc_file,
        temp_coh_file,
        similarity_file,
        shp_count_file,
    )


def _average_or_rename(file_list: list[Path], outfile: Path, output_type: str):
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


def _get_input_dates(
    input_file_list: Sequence[Filename], is_compressed: Sequence[bool], date_fmt: str
) -> list[list[datetime.datetime]]:
    input_dates = [get_dates(f, fmt=date_fmt) for f in input_file_list]
    # For any that aren't compressed, take the first date.
    # this is because the official product name of OPERA/Sentinel1 has both
    # "acquisition_date" ... "generation_date" in the filename
    # For compressed, we want the first 3 dates: (base phase, start, end)
    # TODO: this is a bit hacky, perhaps we can make this some input option
    # so that the user can specify how to get dates from their files (or even
    # directly pass in dates?)
    return [
        dates[:1] if not is_comp else dates[:3]
        for dates, is_comp in zip(input_dates, is_compressed, strict=False)
    ]
