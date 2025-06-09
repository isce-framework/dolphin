from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from opera_utils import group_by_date

from dolphin import PathOrStr, io
from dolphin._log import log_runtime, setup_logging
from dolphin._types import Bbox
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
    """Run the corrections workflow on the displacement outputs.

    Note: Currently this workflow only supports ionospheric corrections
    on OPERA CSLC input datasets.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    correction_options : CorrectionOptions
        Options for the correction workflow.
    timeseries_paths : Sequence[Union[str, Path]]
        Paths to the time series files.
    out_dir : Path, optional
        Output directory, by default current directory.
    reference_point : ReferencePoint, optional
        Reference point for the corrections, by default None.
    debug : bool, optional
        Enable debug logging, by default False.

    Returns
    -------
    CorrectionPaths
        Paths to the correction output files.

    """
    if cfg.log_file is None:
        cfg.log_file = cfg.work_directory / "dolphin.log"
    # Set the logging level for all `dolphin.` modules
    setup_logging(logger_name="dolphin", debug=debug, filename=cfg.log_file)
    logger.debug(cfg.model_dump())

    if len(correction_options.geometry_files) == 0:
        raise ValueError("No geometry files passed to run the corrections workflow")

    grouped_iono_files = parse_ionosphere_files(
        correction_options.ionosphere_files, correction_options._iono_date_fmt
    )
    if not grouped_iono_files:
        raise ValueError("No ionospheric files found for corrections workflow")

    # ##############################################
    # 5. Estimate corrections for each interferogram
    # ##############################################
    iono_paths: list[Path] | None = None

    out_dir = cfg.work_directory / correction_options._atm_directory
    out_dir.mkdir(exist_ok=True)
    grouped_slc_files = group_by_date(cfg.cslc_file_list)

    # Prepare frame geometry files
    geometry_dir = out_dir / "geometry"
    geometry_dir.mkdir(exist_ok=True)

    crs = io.get_raster_crs(timeseries_paths[0])
    epsg = crs.to_epsg()
    assert epsg is not None
    out_bounds = io.get_raster_bounds(timeseries_paths[0])
    frame_geometry_files = prepare_geometry(
        geometry_dir=geometry_dir,
        geo_files=correction_options.geometry_files,
        matching_file=Path(timeseries_paths[0]),
        dem_file=correction_options.dem_file,
        epsg=epsg,
        out_bounds=out_bounds,
        strides=cfg.output_options.strides.model_dump(),
    )

    if reference_point is None:
        from dolphin.timeseries import _read_reference_point

        ref_file = Path(timeseries_paths[0]).parent / "reference_point.txt"
        ref = _read_reference_point(ref_file)
    else:
        ref = ReferencePoint(*reference_point)
    logger.info(
        "Calculating ionospheric corrections for %s files",
        len(timeseries_paths),
    )
    assert timeseries_paths is not None
    iono_paths = estimate_ionospheric_delay(
        ifg_file_list=list(map(Path, timeseries_paths)),
        slc_files=grouped_slc_files,
        tec_files=grouped_iono_files,
        geom_files=frame_geometry_files,
        reference_point=ref,
        output_dir=out_dir,
        epsg=epsg,
        bounds=out_bounds,
    )

    return CorrectionPaths(ionospheric_corrections=iono_paths)


def prepare_geometry(
    geometry_dir: Path,
    geo_files: Sequence[Path],
    matching_file: Path,
    dem_file: Path | None,
    epsg: int,
    out_bounds: Bbox,
    strides: Mapping[str, int] | None = None,
) -> dict[str, Path]:
    """Prepare geometry files.

    Parameters
    ----------
    geometry_dir : Path
        Output directory for geometry files.
    geo_files : list[Path]
        list of geometry files.
    matching_file : Path
        Matching file.
    dem_file : Optional[Path]
        DEM file.
    epsg : int
        EPSG code.
    out_bounds : Bbox
        Output bounds.
    strides : Dict[str, int], optional
        Strides for resampling, by default {"x": 1, "y": 1}.

    Returns
    -------
    Dict[str, Path]
        Dictionary of prepared geometry files.

    """
    from dolphin import stitching
    from dolphin.io import DEFAULT_TIFF_OPTIONS, format_nc_filename

    if strides is None:
        strides = {"x": 1, "y": 1}
    geometry_dir.mkdir(exist_ok=True)

    stitched_geo_list = {}

    if geo_files[0].name.endswith(".h5"):
        # ISCE3 geocoded SLCs
        datasets = ["los_east", "los_north", "layover_shadow_mask"]
        nodatas = [0, 0, 127]

        for nodata, ds_name in zip(nodatas, datasets):
            outfile = geometry_dir / f"{ds_name}.tif"
            logger.info(f"Creating {outfile}")
            stitched_geo_list[ds_name] = outfile
            ds_path = f"/data/{ds_name}"
            cur_files = [format_nc_filename(f, ds_name=ds_path) for f in geo_files]

            if ds_name not in "layover_shadow_mask":
                options = (*DEFAULT_TIFF_OPTIONS, "NBITS=16")
            else:
                options = DEFAULT_TIFF_OPTIONS
            stitching.merge_images(
                cur_files,
                outfile=outfile,
                driver="GTiff",
                out_bounds=out_bounds,
                out_bounds_epsg=epsg,
                in_nodata=nodata,
                out_nodata=nodata,
                target_aligned_pixels=True,
                strides=strides,
                resample_alg="nearest",
                overwrite=False,
                options=options,
            )

        if dem_file:
            height_file = geometry_dir / "height.tif"
            stitched_geo_list["height"] = height_file
            if not height_file.exists():
                logger.info(f"Creating {height_file}")
                stitching.warp_to_match(
                    input_file=dem_file,
                    match_file=matching_file,
                    output_file=height_file,
                    resample_alg="cubic",
                )
    else:
        # ISCE2 radar coordinates
        dsets = {
            "hgt.rdr": "height",
            "incLocal.rdr": "incidence_angle",
            "lat.rdr": "latitude",
            "lon.rdr": "longitude",
        }

        for geo_file in geo_files:
            if geo_file.stem in dsets:
                out_name = dsets[geo_file.stem]
            elif geo_file.name in dsets:
                out_name = dsets[geo_file.name]
                continue

            out_file = geometry_dir / (out_name + ".tif")
            stitched_geo_list[out_name] = out_file
            logger.info(f"Creating {out_file}")

            stitching.warp_to_match(
                input_file=geo_file,
                match_file=matching_file,
                output_file=out_file,
                resample_alg="cubic",
            )

    return stitched_geo_list
