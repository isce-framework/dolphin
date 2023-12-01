"""Estimate wrapped phase using batches of ministacks.

References
----------
    [1] Ansari, H., De Zan, F., & Bamler, R. (2017). Sequential estimator: Toward
    efficient InSAR time series analysis. IEEE Transactions on Geoscience and
    Remote Sensing, 55(10), 5637-5652.
"""

from __future__ import annotations

from itertools import chain
from os import fspath
from pathlib import Path
from typing import Optional

from osgeo_utils import gdal_calc

from dolphin import io
from dolphin._log import get_log
from dolphin._readers import VRTStack
from dolphin._types import Filename
from dolphin.stack import MiniStackPlanner

from .config import ShpMethod
from .single import run_wrapped_phase_single

logger = get_log(__name__)

__all__ = ["run_wrapped_phase_sequential"]


def run_wrapped_phase_sequential(
    *,
    slc_vrt_file: Filename,
    ministack_planner: MiniStackPlanner,
    ministack_size: int,
    half_window: dict,
    strides: dict = {"x": 1, "y": 1},
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int],
    use_evd: bool = False,
    beta: float = 0.01,
    block_shape: tuple[int, int] = (512, 512),
    n_workers: int = 1,
    gpu_enabled: bool = True,
) -> tuple[list[Path], list[Path], Path]:
    """Estimate wrapped phase using batches of ministacks."""
    output_folder = ministack_planner.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    ministacks = ministack_planner.plan(ministack_size)

    v_all = VRTStack.from_vrt_file(slc_vrt_file)
    logger.info(
        f"Full file range for {v_all}: from {v_all.file_list[0]} to {v_all.file_list[-1]}"
    )
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Number of ministacks of size {ministack_size}: {len(ministacks)}")

    if shp_nslc is None:
        shp_nslc = v_all.shape[0]

    # list where each item is [output_slc_files] from a ministack
    output_slc_files: list[list] = []
    # Each item is the temp_coh file from a ministack
    temp_coh_files: list[Path] = []

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

            # Currently: we are always using the first SLC as the reference,
            # even if this is a compressed SLC.
            # Will need to change this if we want to accomodate the original
            # Sequential Estimator+Datum Adjustment method.
            reference_idx = 0
            run_wrapped_phase_single(
                slc_vrt_file=cur_vrt,
                ministack=ministack,
                output_folder=cur_output_folder,
                half_window=half_window,
                strides=strides,
                reference_idx=reference_idx,
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
                n_workers=n_workers,
                gpu_enabled=gpu_enabled,
            )

        cur_output_files, cur_comp_slc_file, temp_coh_file = _get_outputs_from_folder(
            cur_output_folder
        )
        output_slc_files.append(cur_output_files)
        temp_coh_files.append(temp_coh_file)

    ##############################################

    # Average the temporal coherence files in each ministack
    output_temp_coh_file = output_folder / "temporal_coherence_average.tif"
    # we can pass the list of files to gdal_calc, which interprets it
    # as a multi-band file
    if len(temp_coh_files) > 1:
        logger.info(f"Averaging temporal coherence files into: {output_temp_coh_file}")
        gdal_calc.Calc(
            NoDataValue=0,
            format="GTiff",
            outfile=fspath(output_temp_coh_file),
            type="Float32",
            quiet=True,
            overwrite=True,
            creation_options=io.DEFAULT_TIFF_OPTIONS,
            A=temp_coh_files,
            calc="numpy.nanmean(A, axis=0)",
        )
    else:
        temp_coh_files[0].rename(output_temp_coh_file)

    # Combine the separate SLC output lists into a single list
    all_slc_files = list(chain.from_iterable(output_slc_files))
    all_comp_slc_files = [ms.get_compressed_slc_info().path for ms in ministacks]

    pl_outputs = []
    for slc_fname in all_slc_files:
        slc_fname.rename(output_folder / slc_fname.name)
        pl_outputs.append(output_folder / slc_fname.name)

    comp_outputs = []
    for p in all_comp_slc_files:
        p.rename(output_folder / p.name)
        comp_outputs.append(output_folder / p.name)

    return pl_outputs, comp_outputs, output_temp_coh_file


def _get_outputs_from_folder(output_folder: Path):
    cur_output_files = sorted(output_folder.glob("2*.slc.tif"))
    cur_comp_slc_file = next(output_folder.glob("compressed_*"))
    temp_coh_file = next(output_folder.glob("temporal_coherence_*"))
    return cur_output_files, cur_comp_slc_file, temp_coh_file
