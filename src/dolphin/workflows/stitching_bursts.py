"""Stitch burst interferograms (optional) and unwrap them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from shapely import from_wkt

from dolphin import stitching
from dolphin._log import log_runtime
from dolphin._overviews import ImageType, create_image_overviews, create_overviews
from dolphin._types import Bbox
from dolphin.interferogram import estimate_interferometric_correlations
from dolphin.io import EXTRA_COMPRESSED_TIFF_OPTIONS, repack_raster

from .config import OutputOptions

logger = logging.getLogger("dolphin")


@dataclass
class StitchedOutputs:
    """Output rasters from stitching step."""

    ifg_paths: list[Path]
    """List of Paths to the stitched interferograms."""
    interferometric_corr_paths: list[Path]
    """List of Paths to interferometric correlation files created."""
    temp_coh_files: list[Path]
    """List of paths to temporal correlation file created."""
    shp_count_files: list[Path]
    """List of paths to SHP count files created."""
    similarity_files: list[Path]
    """List of paths to cosine similarity files created."""
    crlb_paths: list[Path]
    """List of Paths to Cramer Rao Lower Bound (CRLB) files created."""
    closure_phase_files: list[Path]
    """List of Paths to closure phase files created."""
    ps_file: Path
    """Path to ps mask file created."""
    amp_dispersion_file: Path
    """Path to amplitude dispersion file created."""


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    temp_coh_file_list: Sequence[Path],
    ps_file_list: Sequence[Path],
    crlb_file_list: Sequence[Path],
    closure_phase_file_list: Sequence[Path],
    amp_dispersion_list: Sequence[Path],
    shp_count_file_list: Sequence[Path],
    similarity_file_list: Sequence[Path],
    stitched_ifg_dir: Path,
    output_options: OutputOptions,
    file_date_fmt: str = "%Y%m%d",
    corr_window_size: tuple[int, int] = (11, 11),
    num_workers: int = 3,
) -> StitchedOutputs:
    """Stitch together spatial subsets from phase linking.

    For Sentinel-1 processing, these can be burst-wise interferograms.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence of interferograms files to stitch.
    temp_coh_file_list : Sequence[Path]
        Sequence of paths to the temporal coherence files.
    ps_file_list : Sequence[Path]
        Sequence of paths to the (looked) ps mask files.
    crlb_file_list : Sequence[Path]
        Sequence of paths to the (looked) Cramer Rao Lower Bound (CRLB) files.
    closure_phase_file_list : Sequence[Path]
        Sequence of paths to the (looked) closure phase files.
    amp_dispersion_list : Sequence[Path]
        Sequence of paths to the (looked) amplitude dispersion files.
    shp_count_file_list : Sequence[Path]
        Sequence of paths to the SHP counts files.
    similarity_file_list : Sequence[Path]
        Sequence of paths to the spatial phase cosine similarity files.
    stitched_ifg_dir : Path
        Location to store the output stitched ifgs and correlations
    output_options : OutputOptions
        [`UnwrapWorkflow`][dolphin.workflows.config.OutputOptions] object
        for with parameters for the input/output options
    file_date_fmt : str
        Format of dates contained in filenames.
        default = "%Y%m%d"
    corr_window_size : tuple[int, int]
        Size of moving window (rows, cols) to use for estimating correlation.
        Default = (11, 11)
    num_workers : int
        Number of threads to use for stitching in parallel.
        Default = 3

    Returns
    -------
    stitched_outputs : StitchedOutputs
        [`StitchedOutputs`][dolphin.workflows.stitching_bursts.StitchedOutputs] object
        containing the output stitched ifgs and correlations

    """
    stitched_ifg_dir.mkdir(exist_ok=True, parents=True)
    # Also preps for snaphu, which needs binary format with no nans
    logger.info("Stitching interferograms by date.")
    if output_options.bounds:
        out_bounds = Bbox(*output_options.bounds) if output_options.bounds else None
    elif output_options.bounds_wkt:
        out_bounds = Bbox(*from_wkt(output_options.bounds_wkt).bounds)
    else:
        out_bounds = None

    date_to_ifg_path = stitching.merge_by_date(
        image_file_list=ifg_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_suffix=".int.tif",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        num_workers=num_workers,
    )
    stitched_ifg_paths = list(date_to_ifg_path.values())

    # Estimate the interferometric correlation from the stitched interferogram
    interferometric_corr_paths = estimate_interferometric_correlations(
        stitched_ifg_paths,
        window_size=corr_window_size,
        num_workers=num_workers,
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
    )

    # Stitch the temporal coherence files by date
    date_to_temp_coh_path = stitching.merge_by_date(
        image_file_list=temp_coh_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_prefix="auto",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        num_workers=num_workers,
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
    )
    stitched_temp_coh_files = list(date_to_temp_coh_path.values())

    date_to_shp_count_path = stitching.merge_by_date(
        image_file_list=shp_count_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_prefix="auto",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        num_workers=num_workers,
    )
    stitched_shp_count_files = list(date_to_shp_count_path.values())

    date_to_similarity_path = stitching.merge_by_date(
        image_file_list=similarity_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_prefix="auto",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        resample_alg="nearest",
        num_workers=num_workers,
        # options=EXTRA_COMPRESSED_TIFF_OPTIONS,
    )
    stitched_similarity_files = list(date_to_similarity_path.values())

    # Stitch the looked PS files
    stitched_ps_file = stitched_ifg_dir / "ps_mask_looked.tif"
    if not stitched_ps_file.exists():
        stitching.merge_images(
            ps_file_list,
            outfile=stitched_ps_file,
            out_nodata=255,
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
            dest_epsg=output_options.epsg,
        )

    # Stitch the CRLB estimate files
    date_to_crlb_path = stitching.merge_by_date(
        image_file_list=crlb_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_suffix=".tif",
        output_prefix="crlb_",
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        num_workers=num_workers,
    )
    stitched_crlb_files = list(date_to_crlb_path.values())

    # Stitch the closure phase files
    date_to_closure_phase_path = stitching.merge_by_date(
        image_file_list=closure_phase_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_prefix="closure_phase_",
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        dest_epsg=output_options.epsg,
        num_workers=num_workers,
    )
    stitched_closure_phase_files = list(date_to_closure_phase_path.values())

    # Stitch the amp dispersion files
    stitched_amp_disp_file = stitched_ifg_dir / "amp_dispersion_looked.tif"
    if not stitched_amp_disp_file.exists():
        stitching.merge_images(
            amp_dispersion_list,
            outfile=stitched_amp_disp_file,
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
            dest_epsg=output_options.epsg,
        )
        repack_raster(stitched_amp_disp_file, keep_bits=10)

    if output_options.add_overviews:
        logger.info("Creating overviews for stitched images")
        create_overviews(stitched_ifg_paths, image_type=ImageType.INTERFEROGRAM)
        create_overviews(interferometric_corr_paths, image_type=ImageType.CORRELATION)
        create_overviews(stitched_temp_coh_files, image_type=ImageType.CORRELATION)
        create_overviews(stitched_shp_count_files, image_type=ImageType.PS)
        create_overviews(stitched_similarity_files, image_type=ImageType.CORRELATION)
        create_image_overviews(stitched_ps_file, image_type=ImageType.PS)
        create_image_overviews(stitched_amp_disp_file, image_type=ImageType.CORRELATION)

    return StitchedOutputs(
        stitched_ifg_paths,
        interferometric_corr_paths,
        stitched_temp_coh_files,
        stitched_shp_count_files,
        stitched_similarity_files,
        stitched_crlb_files,
        stitched_closure_phase_files,
        stitched_ps_file,
        stitched_amp_disp_file,
    )
