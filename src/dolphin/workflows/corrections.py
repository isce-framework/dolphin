from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from opera_utils import group_by_date

from dolphin import PathOrStr, io, utils
from dolphin._log import log_runtime, setup_logging
from dolphin.atmosphere import estimate_ionospheric_delay
from dolphin.timeseries import ReferencePoint
from dolphin.workflows import CorrectionOptions

from ._utils import parse_ionosphere_files
from .config import DisplacementWorkflow

logger = logging.getLogger(__name__)


@dataclass
class CorrectionPaths:
    """Output files of the Corrections workflow."""

    ionospheric_corrections: list[Path] | None


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    correction_options: CorrectionOptions,
    timeseries_paths: Sequence[PathOrStr],
    out_dir: Path = Path(),
    reference_point: ReferencePoint | None = None,
    debug: bool = False,
) -> CorrectionPaths:
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
    for logger_name in ["dolphin", "spurt"]:
        setup_logging(logger_name=logger_name, debug=debug, filename=cfg.log_file)
    logger.debug(cfg.model_dump())

    grouped_iono_files = parse_ionosphere_files(
        correction_options.ionosphere_files, correction_options._iono_date_fmt
    )
    if len(correction_options.geometry_files) > 0:
        raise ValueError("No geometry files passed to run the corrections workflow")

    # ##############################################
    # 5. Estimate corrections for each interferogram
    # ##############################################
    tropo_paths: list[Path] | None = None
    iono_paths: list[Path] | None = None

    out_dir = cfg.work_directory / correction_options._atm_directory
    out_dir.mkdir(exist_ok=True)
    grouped_slc_files = group_by_date(cfg.cslc_file_list)

    # Prepare frame geometry files
    geometry_dir = out_dir / "geometry"
    geometry_dir.mkdir(exist_ok=True)

    crs = io.get_raster_crs(timeseries_paths[0])
    epsg = crs.to_epsg()
    out_bounds = io.get_raster_bounds(timeseries_paths[0])
    frame_geometry_files = utils.prepare_geometry(
        geometry_dir=geometry_dir,
        geo_files=correction_options.geometry_files,
        matching_file=timeseries_paths[0],
        dem_file=correction_options.dem_file,
        epsg=epsg,
        out_bounds=out_bounds,
        strides=cfg.output_options.strides,
    )

    # Troposphere
    if "height" not in frame_geometry_files:
        logger.warning(
            "DEM file is not given, skip estimating tropospheric corrections."
        )
    else:
        if correction_options.troposphere_files:
            from dolphin.atmosphere import estimate_tropospheric_delay

            assert timeseries_paths is not None
            logger.info(
                "Calculating tropospheric corrections for %s files.",
                len(timeseries_paths),
            )
            tropo_paths = estimate_tropospheric_delay(
                ifg_file_list=timeseries_paths,
                troposphere_files=correction_options.troposphere_files,
                file_date_fmt=correction_options.tropo_date_fmt,
                slc_files=grouped_slc_files,
                geom_files=frame_geometry_files,
                reference_point=reference_point,
                output_dir=out_dir,
                tropo_model=correction_options.tropo_model,
                tropo_delay_type=correction_options.tropo_delay_type,
                epsg=epsg,
                bounds=out_bounds,
            )
        else:
            logger.info("No weather model, skip tropospheric correction.")

    # Ionosphere
    if not grouped_iono_files:
        raise ValueError()

    logger.info(
        "Calculating ionospheric corrections for %s files",
        len(timeseries_paths),
    )
    assert timeseries_paths is not None
    iono_paths = estimate_ionospheric_delay(
        ifg_file_list=timeseries_paths,
        slc_files=grouped_slc_files,
        tec_files=grouped_iono_files,
        geom_files=frame_geometry_files,
        reference_point=reference_point,
        output_dir=out_dir,
        epsg=epsg,
        bounds=out_bounds,
    )

    return CorrectionPaths(
        ionospheric_corrections=iono_paths, tropospheric_corrections=tropo_paths
    )
