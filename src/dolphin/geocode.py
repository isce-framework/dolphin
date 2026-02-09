"""Geocode rasters from radar to geographic coordinates using geolocation arrays."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from osgeo import gdal, osr

from dolphin import io

gdal.UseExceptions()

logger = logging.getLogger("dolphin")

__all__ = ["geocode_with_geolocation_arrays", "run"]

DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=lzw",
    "BIGTIFF=IF_SAFER",
    "TILED=yes",
    "INTERLEAVE=band",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)


def geocode_with_geolocation_arrays(
    input_file: Path | str,
    lat_file: Path | str,
    lon_file: Path | str,
    output_file: Path | str | None = None,
    mask_file: Path | str | None = None,
    output_srs: str | int | None = None,
    spacing: tuple[float, float] | float | None = None,
    output_format: str = "GTiff",
    bounds: tuple[float, float, float, float] | None = None,
    resampling_method: str = "near",
    strides: tuple[int, int] | None = None,
    creation_options: Sequence[str] = DEFAULT_TIFF_OPTIONS,
) -> Path:
    """Geocode a swath file using latitude and longitude geolocation arrays.

    Uses GDAL's geolocation array warping to transform radar/swath geometry
    data to a geographic or projected coordinate system.

    Parameters
    ----------
    input_file : Path or str
        Path to the input swath file to geocode.
    lat_file : Path or str
        Path to file containing per-pixel latitude values (e.g., from ISCE topo).
    lon_file : Path or str
        Path to file containing per-pixel longitude values.
    output_file : Path or str, optional
        Path for the output geocoded file. Default is ``<input>.geo.<ext>``.
    mask_file : Path or str, optional
        Path to a mask raster in the same geometry as `input_file`.
        Convention: 0 = invalid, nonzero = valid (SNAPHU/ISCE convention).
        If the mask is at full resolution and `strides` are given, it will be
        subsampled to match the input.
    output_srs : str or int, optional
        Output spatial reference system as EPSG code (int) or WKT/proj4 string.
        If None, defaults to EPSG:4326 (WGS84 geographic).
    spacing : tuple of float or float, optional
        Output pixel spacing as ``(x_res, y_res)`` or a single value for both.
        If None, GDAL determines spacing automatically.
    output_format : str, default="GTiff"
        GDAL driver name for the output format.
    bounds : tuple of float, optional
        Output bounds as ``(xmin, ymin, xmax, ymax)`` in output SRS coordinates.
    resampling_method : str, default="near"
        GDAL resampling algorithm (e.g., "near", "bilinear", "cubic").
    strides : tuple of int, optional
        Strides as ``(row_stride, col_stride)`` indicating the additional spacing
        in the `input_file` compared to `lat_file` and `lon_file`.
        For example, if multilooked interferograms were created with
        2 range (col) and 3 azimuth (row) strides, set ``strides=(3, 2)``.
    creation_options : list of str, optional
        GDAL creation options for the output file.

    Returns
    -------
    Path
        Path to the created geocoded file.

    Notes
    -----
    The geolocation arrays (lat/lon files) are assumed to be in WGS84 (EPSG:4326),
    which is standard for ISCE2/ISCE3 topo outputs.

    When using `strides`, the lat/lon arrays are subsampled via VRT to match
    the input resolution before warping.

    """
    input_file = Path(input_file)
    lat_file = Path(lat_file)
    lon_file = Path(lon_file)
    mask_file = Path(mask_file) if mask_file is not None else None

    if output_file is None:
        output_file = input_file.with_suffix(f".geo{input_file.suffix}")
    output_file = Path(output_file)

    # Parse spacing input
    if spacing is None:
        x_res, y_res = None, None
    elif isinstance(spacing, int | float):
        x_res = y_res = float(spacing)
    else:
        x_res, y_res = spacing

    # Parse strides
    if strides is None:
        row_stride, col_stride = 1, 1
    else:
        row_stride, col_stride = strides

    # VRT XML template for source bands
    source_xml_template = """\
    <SimpleSource>
      <SourceFilename>{filename}</SourceFilename>
      <SourceBand>{band}</SourceBand>
    </SimpleSource>"""

    input_ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    assert input_ds is not None, f"Could not open {input_file}"

    # Check if mask needs subsampling to match input (when strides are used)
    mask_needs_subsample = False
    if mask_file is not None:
        mask_ds = gdal.Open(str(mask_file), gdal.GA_ReadOnly)
        assert mask_ds is not None, f"Could not open mask file {mask_file}"

        mask_matches = (
            mask_ds.RasterXSize == input_ds.RasterXSize
            and mask_ds.RasterYSize == input_ds.RasterYSize
        )
        # Check if mask matches input*strides (full-res mask for multilooked input)
        mask_matches_with_strides = (
            mask_ds.RasterXSize == input_ds.RasterXSize * col_stride
            and mask_ds.RasterYSize == input_ds.RasterYSize * row_stride
        )

        if not mask_matches and not mask_matches_with_strides:
            expected_strided = (
                input_ds.RasterYSize * row_stride,
                input_ds.RasterXSize * col_stride,
            )
            msg = (
                f"Mask shape {mask_ds.RasterYSize, mask_ds.RasterXSize} must match "
                f"input {input_ds.RasterYSize, input_ds.RasterXSize} or "
                f"input*strides {expected_strided}"
            )
            raise ValueError(msg)

        mask_needs_subsample = mask_matches_with_strides and not mask_matches
        mask_ds = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        temp_vrt_path = tmp_path / "geocode.vrt"
        mask_alpha_file = None

        if mask_file is not None:
            # If mask is at full resolution but input is multilooked, subsample
            mask_to_use = mask_file
            if mask_needs_subsample:
                subsampled_mask = tmp_path / "mask_subsampled.vrt"
                _create_subsampled_vrt(
                    mask_file, subsampled_mask, row_stride, col_stride
                )
                mask_to_use = subsampled_mask

            # Convert mask to alpha band: 255=valid, 0=invalid
            # Assumes 0=invalid, nonzero=valid (SNAPHU/ISCE convention)
            mask_alpha_file = tmp_path / "mask_alpha.tif"
            _create_alpha_from_mask(mask_to_use, mask_alpha_file)

        # If strides > 1, subsample lat/lon arrays to match input size
        lat_to_use: Path | str = lat_file
        lon_to_use: Path | str = lon_file
        if row_stride > 1 or col_stride > 1:
            subsampled_lat = tmp_path / "lat_subsampled.vrt"
            subsampled_lon = tmp_path / "lon_subsampled.vrt"
            _create_subsampled_vrt(lat_file, subsampled_lat, row_stride, col_stride)
            _create_subsampled_vrt(lon_file, subsampled_lon, row_stride, col_stride)
            lat_to_use = subsampled_lat
            lon_to_use = subsampled_lon

        # Create VRT with geolocation metadata
        driver = gdal.GetDriverByName("VRT")
        vrt_ds = driver.Create(
            str(temp_vrt_path), input_ds.RasterXSize, input_ds.RasterYSize, 0
        )

        # Copy bands from input to VRT
        nodata_vals = []
        for band_idx in range(input_ds.RasterCount):
            band = input_ds.GetRasterBand(band_idx + 1)
            vrt_ds.AddBand(band.DataType)
            nodata_vals.append(band.GetNoDataValue())
            source_xml = source_xml_template.format(
                filename=str(input_file), band=band_idx + 1
            )
            vrt_ds.GetRasterBand(band_idx + 1).SetMetadata(
                {"source_0": source_xml}, "vrt_sources"
            )

        if mask_alpha_file is not None:
            alpha_band_index = input_ds.RasterCount + 1
            vrt_ds.AddBand(gdal.GDT_Byte)
            alpha_band = vrt_ds.GetRasterBand(alpha_band_index)
            alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
            source_xml = source_xml_template.format(
                filename=str(mask_alpha_file), band=1
            )
            alpha_band.SetMetadata({"source_0": source_xml}, "vrt_sources")

        # Geolocation arrays are in WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Set geolocation metadata. When strides are used, we've already subsampled
        # the lat/lon arrays to match the input, so PIXEL_STEP=LINE_STEP=1.
        vrt_ds.SetMetadata(
            {
                "SRS": srs.ExportToWkt(),
                "X_DATASET": str(lon_to_use),
                "X_BAND": "1",
                "Y_DATASET": str(lat_to_use),
                "Y_BAND": "1",
                "PIXEL_OFFSET": "0",
                "LINE_OFFSET": "0",
                "PIXEL_STEP": "1",
                "LINE_STEP": "1",
            },
            "GEOLOCATION",
        )
        for i in range(len(nodata_vals)):
            if nodata_vals[i] is not None:
                vrt_ds.GetRasterBand(i + 1).SetNoDataValue(nodata_vals[i])

        # Flush before warping
        vrt_ds = None
        input_ds = None

        warp_options = gdal.WarpOptions(
            format=output_format,
            xRes=x_res,
            yRes=y_res,
            dstSRS=output_srs,
            outputBounds=bounds,
            resampleAlg=resampling_method,
            geoloc=True,
            srcAlpha=mask_alpha_file is not None,
            dstNodata=np.nan,
            creationOptions=creation_options,
        )
        gdal.Warp(str(output_file), str(temp_vrt_path), options=warp_options)

    return output_file


def _create_subsampled_vrt(
    input_file: Path,
    output_file: Path,
    row_stride: int,
    col_stride: int,
) -> Path:
    """Create a VRT that subsamples a raster by the given strides."""
    input_ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    assert input_ds is not None, f"Could not open {input_file}"

    in_width = input_ds.RasterXSize
    in_height = input_ds.RasterYSize
    out_width = in_width // col_stride
    out_height = in_height // row_stride

    band = input_ds.GetRasterBand(1)
    dtype_name = gdal.GetDataTypeName(band.DataType)
    nodata = band.GetNoDataValue()
    nodata_xml = f"<NoDataValue>{nodata}</NoDataValue>" if nodata is not None else ""
    input_ds = None

    # VRT with SrcRect covering full input, DstRect smaller -> GDAL subsamples
    vrt_xml = f"""\
<VRTDataset rasterXSize="{out_width}" rasterYSize="{out_height}">
  <VRTRasterBand dataType="{dtype_name}" band="1">
    {nodata_xml}
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{input_file}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{in_width}" ySize="{in_height}"/>
      <DstRect xOff="0" yOff="0" xSize="{out_width}" ySize="{out_height}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""

    output_file.write_text(vrt_xml)
    return output_file


def _create_alpha_from_mask(
    mask_file: Path | str,
    alpha_file: Path | str,
) -> None:
    """Create an alpha band (255=valid, 0=invalid) from a binary mask.

    Assumes mask convention where 0 = invalid, nonzero = valid.
    """
    mask = io.load_gdal(mask_file).astype(bool)
    alpha = np.where(mask, np.uint8(255), np.uint8(0))
    io.write_arr(arr=alpha, output_name=alpha_file, like_filename=mask_file)


def _geocode_one(
    in_out: tuple[Path, Path],
    *,
    lat_file: Path,
    lon_file: Path,
    mask_file: Path | None,
    output_srs: str | int | None,
    spacing: tuple[float, float] | None,
    strides: tuple[int, int] | None,
    resampling_method: str,
    creation_options: Sequence[str],
) -> Path:
    """Geocode a single file (module-level for pickling)."""
    in_path, out_path = in_out
    geocode_with_geolocation_arrays(
        input_file=in_path,
        lat_file=lat_file,
        lon_file=lon_file,
        output_file=out_path,
        mask_file=mask_file,
        output_srs=output_srs,
        spacing=spacing,
        strides=strides,
        resampling_method=resampling_method,
        creation_options=creation_options,
    )
    return out_path


def run(
    input_files: Annotated[
        list[Path],
        tyro.conf.arg(
            aliases=["-i", "--input"],
            help="Input file(s) to geocode. Can be specified multiple times.",
        ),
    ],
    output: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-o"],
            help=(
                "Output path. For single input, this is the output file path. "
                "For multiple inputs, this should be a directory (files will be "
                "named <input>.geo.tif). If not provided, outputs are written "
                "alongside inputs."
            ),
        ),
    ] = None,
    geometry_dir: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-g", "--geometry"],
            help=(
                "Geometry directory containing y.tif (lat) and x.tif (lon). "
                "Shorthand for --lat <dir>/y.tif --lon <dir>/x.tif."
            ),
        ),
    ] = None,
    lat_file: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["--lat"],
            help="Path to latitude geolocation file.",
        ),
    ] = None,
    lon_file: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["--lon"],
            help="Path to longitude geolocation file.",
        ),
    ] = None,
    mask: Annotated[
        Path | None,
        tyro.conf.arg(
            help=(
                "Mask file to apply during geocoding (0=invalid, nonzero=valid)."
                " Should match input resolution, or full resolution if strides"
                " are read from --config."
            ),
        ),
    ] = None,
    output_srs: Annotated[
        str | int | None,
        tyro.conf.arg(
            aliases=["--srs"],
            help=(
                "Output spatial reference system as EPSG code (e.g., 32610 for UTM"
                " 10N) or proj4/WKT string. Defaults to EPSG:4326 (WGS84 lat/lon)."
            ),
        ),
    ] = None,
    spacing: Annotated[
        float | None,
        tyro.conf.arg(
            aliases=["-s"],
            help=(
                "Output pixel spacing in output SRS units (degrees for EPSG:4326, "
                "meters for UTM). If not provided, GDAL determines automatically."
            ),
        ),
    ] = None,
    config: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-c"],
            help=(
                "Path to dolphin_config.yaml. Reads"
                " output_options.strides to determine the decimation factor"
                " of input files relative to the lat/lon geolocation files."
            ),
        ),
    ] = None,
    resampling_method: str = "near",
    creation_options: Sequence[str] = DEFAULT_TIFF_OPTIONS,
    max_workers: Annotated[
        int | None,
        tyro.conf.arg(
            aliases=["-j", "--jobs"],
            help="Max parallel workers. Defaults to number of CPUs.",
        ),
    ] = None,
) -> list[Path]:
    r"""Geocode rasters using latitude/longitude geolocation arrays.

    Transforms rasters from radar/swath geometry to geographic coordinates
    using per-pixel lat/lon arrays (e.g., from ISCE2/ISCE3 topo).

    Examples
    --------
    Single file with geometry directory:

        dolphin geocode -i interferogram.tif -g geometry/

    Multiple files to output directory:

        dolphin geocode -i ifg.tif -i coherence.tif -g geometry/ -o geocoded/

    With mask and strides from dolphin config (for multilooked outputs):

        dolphin geocode -i ifg.tif -g geometry/ --mask mask.tif -c dolphin_config.yaml

    Output to UTM with 30m spacing:

        dolphin geocode -i ifg.tif -g geometry/ --srs 32610 -s 30

    Using explicit lat/lon files:

        dolphin geocode -i ifg.tif --lat y.tif --lon x.tif

    """
    # Resolve lat/lon files
    if geometry_dir is not None:
        geometry_dir = Path(geometry_dir)
        lat_file = geometry_dir / "y.tif"
        lon_file = geometry_dir / "x.tif"

    if lat_file is None or lon_file is None:
        msg = "Must provide either --geometry or both --lat and --lon"
        raise ValueError(msg)

    lat_file = Path(lat_file)
    lon_file = Path(lon_file)
    assert lat_file.exists(), f"Lat file not found: {lat_file}"
    assert lon_file.exists(), f"Lon file not found: {lon_file}"

    # Read strides from dolphin config
    parsed_strides: tuple[int, int] | None = None
    if config is not None:
        from dolphin.workflows.config import DisplacementWorkflow

        cfg = DisplacementWorkflow.from_yaml(config)
        sy = cfg.output_options.strides.y
        sx = cfg.output_options.strides.x
        if sy != 1 or sx != 1:
            parsed_strides = (sy, sx)
            logger.info("Using strides from config: (%d, %d)", sy, sx)

    # Parse spacing
    parsed_spacing: tuple[float, float] | None = None
    if spacing is not None:
        parsed_spacing = (spacing, spacing)

    # Determine output paths
    multiple_inputs = len(input_files) > 1
    output_is_dir = output is not None and (output.is_dir() or multiple_inputs)

    if multiple_inputs and output is not None:
        output.mkdir(parents=True, exist_ok=True)

    # Build list of (input, output) pairs
    io_pairs: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    for in_file in input_files:
        in_path = Path(in_file)
        if output is None:
            out_path = in_path.with_suffix(f".geo{in_path.suffix}")
        elif output_is_dir:
            out_path = output / f"{in_path.stem}.geo{in_path.suffix}"
        else:
            out_path = output

        # Skip if output exists and is newer than input
        if out_path.exists() and out_path.stat().st_mtime >= in_path.stat().st_mtime:
            logger.debug("Skipping %s (already exists)", out_path.name)
            skipped.append(out_path)
        else:
            io_pairs.append((in_path, out_path))

    if not io_pairs:
        logger.info("All files already geocoded, skipping")
        return skipped

    # Geocode in parallel
    n_workers = max_workers or os.cpu_count() or 1
    n_workers = min(n_workers, len(io_pairs))
    logger.info("Geocoding %d file(s) with %d workers", len(io_pairs), n_workers)

    worker = partial(
        _geocode_one,
        lat_file=lat_file,
        lon_file=lon_file,
        mask_file=mask,
        output_srs=output_srs,
        spacing=parsed_spacing,
        strides=parsed_strides,
        resampling_method=resampling_method,
        creation_options=creation_options,
    )

    output_files: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, pair): pair[1] for pair in io_pairs}
        for future in as_completed(futures):
            out_path = future.result()
            logger.info("Completed: %s", out_path.name)
            output_files.append(out_path)

    logger.info("Geocoded %d file(s)", len(output_files))
    return skipped + output_files
