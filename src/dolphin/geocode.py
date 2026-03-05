"""Geocode rasters from radar to geographic coordinates using geolocation arrays.

Supports ISCE3-style geometry (``y.tif``/``x.tif``), ISCE2-style geometry
(``lat.rdr``/``lon.rdr``), and common GAMMA-style geometry naming
(``lat.rdc``/``lon.rdc`` and variants) for per-pixel geolocation arrays.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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

__all__ = ["find_rasters_to_geocode", "geocode_with_geolocation_arrays", "run"]

DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=lzw",
    "BIGTIFF=IF_SAFER",
    "TILED=yes",
    "INTERLEAVE=band",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
)


def geocode_with_gamma_lookup_table(
    input_file: Path | str,
    lookup_file: Path | str,
    dem_par_file: Path | str,
    output_file: Path | str | None = None,
    strides: tuple[int, int] | None = None,
    geometry_dir: Path | None = None,
    resampling_method: str = "bilinear",
    creation_options: Sequence[str] = DEFAULT_TIFF_OPTIONS,
) -> Path:
    """Geocode a radar-grid raster using a GAMMA lookup table.

    Parameters
    ----------
    input_file : Path or str
        Radar-coordinate raster to geocode (e.g., Dolphin velocity in timeseries/).
    lookup_file : Path or str
        GAMMA lookup table file (A->B) as FCOMPLEX binary, e.g. ``lookup_fine``.
    dem_par_file : Path or str
        DEM segment parameter file (e.g. ``dem_seg.par``) defining output map grid.
    output_file : Path or str, optional
        Output geocoded file path. Default is ``<input>.geo.<ext>``.
    strides : tuple of int, optional
        Strides as ``(row_stride, col_stride)`` for multilooked input rasters relative
        to the 1x1 lookup table coordinates.
    resampling_method : str, default="bilinear"
        Interpolation method: ``near``/``nearest`` or ``bilinear``.
    creation_options : list of str, optional
        GDAL creation options for output GeoTIFF.

    Returns
    -------
    Path
        Path to output geocoded raster.

    """
    input_file = Path(input_file)
    lookup_file = Path(lookup_file)
    dem_par_file = Path(dem_par_file)
    if output_file is None:
        output_file = input_file.with_suffix(f".geo{input_file.suffix}")
    output_file = Path(output_file)

    row_stride, col_stride = strides or (1, 1)
    dem_meta = _read_gamma_dem_par(dem_par_file)
    out_width = dem_meta["width"]
    out_height = dem_meta["nlines"]

    input_ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    assert input_ds is not None, f"Could not open {input_file}"
    in_width = input_ds.RasterXSize
    in_height = input_ds.RasterYSize

    # Preferred path: generate lat/lon in radar geometry using GAMMA pt2geo,
    # then use GDAL geoloc warp. This avoids lookup component/order ambiguities.
    try:
        lat_tif, lon_tif = _make_latlon_from_gamma_pt2geo(
            input_file=input_file,
            geometry_dir=geometry_dir or lookup_file.parent,
        )
        input_ds = None
        return geocode_with_geolocation_arrays(
            input_file=input_file,
            lat_file=lat_tif,
            lon_file=lon_tif,
            output_file=output_file,
            strides=(row_stride, col_stride),
            resampling_method=resampling_method,
            creation_options=creation_options,
        )
    except Exception as e:
        logger.warning(
            "pt2geo lat/lon path unavailable, fallback to lookup mode: %s", e
        )

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_file),
        out_width,
        out_height,
        input_ds.RasterCount,
        input_ds.GetRasterBand(1).DataType,
        options=list(creation_options),
    )

    geotransform = (
        dem_meta["corner_lon"],
        dem_meta["post_lon"],
        0.0,
        dem_meta["corner_lat"],
        0.0,
        dem_meta["post_lat"],
    )
    out_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    out_ds.SetProjection(srs.ExportToWkt())

    # Prefer GAMMA's native resampling to avoid orientation/indexing mismatches.
    use_gamma_binary = shutil.which("geocode_back") is not None
    if use_gamma_binary:
        logger.info("Using GAMMA geocode_back for lookup-table geocoding")
    else:
        logger.warning(
            "geocode_back not found in PATH; falling back to internal lookup resampler"
        )
        lookup_raw = np.fromfile(lookup_file, dtype=np.complex64)
        expected = out_width * out_height
        if lookup_raw.size != expected:
            msg = (
                f"Lookup size mismatch for {lookup_file}: got {lookup_raw.size} complex"
                f" samples, expected {expected} from {dem_par_file}"
            )
            raise ValueError(msg)
        lookup = lookup_raw.reshape(out_height, out_width)
        x_coords, y_coords, mode_tag, in_bounds_ratio = (
            _extract_best_lookup_coordinates(
                lookup=lookup,
                in_width=in_width,
                in_height=in_height,
                row_stride=row_stride,
                col_stride=col_stride,
            )
        )
        logger.info(
            "Fallback lookup decode mode=%s (in-bounds %.1f%%)",
            mode_tag,
            100.0 * in_bounds_ratio,
        )

    method = resampling_method.lower()
    for band_idx in range(1, input_ds.RasterCount + 1):
        in_band = input_ds.GetRasterBand(band_idx)
        in_arr = in_band.ReadAsArray()
        nodata = in_band.GetNoDataValue()
        if use_gamma_binary:
            out_arr = _resample_with_geocode_back(
                in_arr=in_arr,
                lookup_file=lookup_file,
                in_width=in_width,
                out_width=out_width,
                out_height=out_height,
                row_stride=row_stride,
                col_stride=col_stride,
                method=method,
            )
        else:
            out_arr = _resample_from_lookup(
                in_arr=in_arr,
                x_coords=x_coords,
                y_coords=y_coords,
                method=method,
                nodata=nodata,
            )
        out_band = out_ds.GetRasterBand(band_idx)
        out_band.WriteArray(out_arr)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

    out_ds.FlushCache()
    out_ds = None
    input_ds = None
    return output_file


def _resample_with_geocode_back(
    *,
    in_arr: np.ndarray,
    lookup_file: Path,
    in_width: int,
    out_width: int,
    out_height: int,
    row_stride: int,
    col_stride: int,
    method: str,
) -> np.ndarray:
    """Resample using GAMMA geocode_back.

    We pass the original 1x1 lookup table and account for multilooking by
    scaling lookup coordinates before calling geocode_back.
    """
    interp_mode = 0 if method in {"near", "nearest", "nearest-neighbor"} else 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        in_bin = tmp / "in.float"
        out_bin = tmp / "out.float"
        lut_scaled = tmp / "lookup_scaled"

        in_arr.astype(np.float32).tofile(in_bin)
        _write_scaled_lookup_file(
            input_lookup=lookup_file,
            output_lookup=lut_scaled,
            out_width=out_width,
            out_height=out_height,
            in_width=in_width,
            in_height=in_arr.shape[0],
            row_stride=row_stride,
            col_stride=col_stride,
        )

        cmd = [
            "geocode_back",
            str(in_bin),
            str(in_width),
            str(lut_scaled),
            str(out_bin),
            str(out_width),
            str(out_height),
            str(interp_mode),
            "0",  # FLOAT input/output
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = np.fromfile(out_bin, dtype=np.float32)
        expected = out_width * out_height
        if out.size != expected:
            msg = (
                f"geocode_back output size mismatch: got {out.size}, expected"
                f" {expected}"
            )
            raise RuntimeError(msg)
        return out.reshape(out_height, out_width)


def _write_scaled_lookup_file(
    *,
    input_lookup: Path,
    output_lookup: Path,
    out_width: int,
    out_height: int,
    in_width: int,
    in_height: int,
    row_stride: int,
    col_stride: int,
) -> None:
    """Scale a 1x1 lookup table for looked products and write FCOMPLEX output."""
    expected = out_width * out_height

    def _score(dtype: np.dtype, swap_components: bool) -> tuple[np.ndarray, float]:
        arr = np.fromfile(input_lookup, dtype=dtype)
        if arr.size != expected:
            msg = f"Lookup size mismatch: got {arr.size}, expected {expected}"
            raise ValueError(msg)
        lut_local = arr.reshape(out_height, out_width)
        if swap_components:
            x = lut_local.imag
            y = lut_local.real
        else:
            x = lut_local.real
            y = lut_local.imag
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return lut_local, 0.0
        x2 = x[finite] / float(col_stride)
        y2 = y[finite] / float(row_stride)
        in_bounds = (
            (x2 >= -0.5)
            & (x2 <= in_width - 0.5)
            & (y2 >= -0.5)
            & (y2 <= in_height - 0.5)
        )
        return lut_local, float(np.count_nonzero(in_bounds)) / in_bounds.size

    candidates: list[tuple[np.ndarray, float, bool, bool]] = []
    for dtype, is_be in ((np.dtype(">c8"), True), (np.dtype("<c8"), False)):
        for swap in (False, True):
            lut_cur, score = _score(dtype, swap)
            candidates.append((lut_cur, score, is_be, swap))

    lut, best_score, use_be, swap_components = max(candidates, key=lambda x: x[1])
    lut = lut.copy()

    if swap_components:
        x_raw = lut.imag.copy()
        y_raw = lut.real.copy()
    else:
        x_raw = lut.real.copy()
        y_raw = lut.imag.copy()

    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw[finite] /= float(col_stride)
    y_raw[finite] /= float(row_stride)

    # geocode_back expects: real = range sample (x), imag = azimuth line (y)
    lut.real = x_raw
    lut.imag = y_raw

    if use_be:
        logger.info(
            "Scaling lookup as big-endian FCOMPLEX (swap_components=%s, score=%.3f)",
            swap_components,
            best_score,
        )
        lut.astype(np.dtype(">c8")).tofile(output_lookup)
    else:
        logger.info(
            "Scaling lookup as little-endian FCOMPLEX (swap_components=%s, score=%.3f)",
            swap_components,
            best_score,
        )
        lut.astype(np.dtype("<c8")).tofile(output_lookup)


def _make_latlon_from_gamma_pt2geo(
    *,
    input_file: Path,
    geometry_dir: Path,
) -> tuple[Path, Path]:
    """Generate full-resolution lat/lon rasters via GAMMA mkgrid + pt2geo."""
    if shutil.which("mkgrid") is None or shutil.which("pt2geo") is None:
        raise RuntimeError("mkgrid/pt2geo not found in PATH")

    diff_par = _find_single_file(geometry_dir, "*.diff_par")
    dem_par = geometry_dir / "dem_seg.par"
    if not dem_par.exists():
        raise FileNotFoundError(f"dem_seg.par not found in {geometry_dir}")

    date_tag = diff_par.stem
    hgt = geometry_dir / f"{date_tag}.hgt"
    if not hgt.exists():
        hgt = _find_single_file(geometry_dir, "*.hgt")

    rslc_par_candidates = [
        geometry_dir / "rslc" / f"{date_tag}.rslc.par",
        geometry_dir.parent / "rslc" / f"{date_tag}.rslc.par",
        geometry_dir.parent / f"{date_tag}.rslc.par",
    ]
    rslc_par = next((p for p in rslc_par_candidates if p.exists()), None)
    if rslc_par is None:
        rslc_par = _find_single_file(geometry_dir.parent / "rslc", "*.rslc.par")

    full_width, full_height = _read_gamma_rslc_size(rslc_par)

    out_dir = input_file.parent / ".gamma_geocode_cache"
    out_dir.mkdir(exist_ok=True)
    lat_tif = out_dir / f"{input_file.stem}.lat.tif"
    lon_tif = out_dir / f"{input_file.stem}.lon.tif"
    if lat_tif.exists() and lon_tif.exists():
        return lat_tif, lon_tif

    plist = out_dir / f"{input_file.stem}.plist"
    plat_lon = out_dir / f"{input_file.stem}.plat_lon"
    phgt = out_dir / f"{input_file.stem}.phgt_wgs84"
    log_file = out_dir / "gamma_pt2geo.log"

    cmd_mkgrid = ["mkgrid", str(plist), str(full_width), str(full_height), "1", "1"]
    with log_file.open("a", encoding="utf-8") as fid:
        subprocess.run(cmd_mkgrid, check=True, stdout=fid, stderr=fid)

    cmd_pt2geo = [
        "pt2geo",
        str(plist),
        "-",
        str(rslc_par),
        "-",
        str(hgt),
        str(dem_par),
        str(diff_par),
        "1",
        "1",
        "-",
        "-",
        str(plat_lon),
        str(phgt),
    ]
    with log_file.open("a", encoding="utf-8") as fid:
        subprocess.run(cmd_pt2geo, check=True, stdout=fid, stderr=fid)

    lon_arr, lat_arr = _read_gamma_plat_lon(
        plat_lon,
        width=full_width,
        height=full_height,
    )
    _write_float32_raster(lat_tif, lat_arr.astype(np.float32))
    _write_float32_raster(lon_tif, lon_arr.astype(np.float32))
    return lat_tif, lon_tif


def _read_gamma_rslc_size(rslc_par: Path) -> tuple[int, int]:
    """Read (range_samples, azimuth_lines) from GAMMA RSLC .par file."""
    vals: dict[str, str] = {}
    for line in rslc_par.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        vals[key.strip()] = value.strip().split()[0]
    return int(vals["range_samples"]), int(vals["azimuth_lines"])


def _write_float32_raster(path: Path, data: np.ndarray) -> None:
    """Write a single-band float32 GeoTIFF without georeferencing metadata."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None


def _read_gamma_plat_lon(
    plat_lon_file: Path,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read GAMMA plat_lon list and return (lon, lat) arrays."""
    n = width * height * 2
    raw_be = np.fromfile(plat_lon_file, dtype=np.dtype(">f8"))
    raw_le = np.fromfile(plat_lon_file, dtype=np.dtype("<f8"))
    if raw_be.size != n and raw_le.size != n:
        msg = (
            f"Unexpected plat_lon size in {plat_lon_file}: {raw_be.size} /"
            f" {raw_le.size}"
        )
        raise ValueError(msg)

    def _score(raw: np.ndarray) -> tuple[np.ndarray, float]:
        arr = raw[:n].reshape(height, width, 2)
        lon = arr[..., 0]
        lat = arr[..., 1]
        valid = (
            np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= -180)
            & (lon <= 180)
            & (lat >= -90)
            & (lat <= 90)
        )
        return arr, float(np.count_nonzero(valid)) / valid.size

    arr_be, score_be = _score(raw_be) if raw_be.size >= n else (None, -1.0)
    arr_le, score_le = _score(raw_le) if raw_le.size >= n else (None, -1.0)
    arr = arr_be if score_be >= score_le else arr_le
    assert arr is not None
    return arr[..., 0], arr[..., 1]


def _find_single_file(directory: Path, pattern: str) -> Path:
    """Find exactly one matching file or return the first sorted match."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return matches[0]


def _read_gamma_dem_par(dem_par_file: Path) -> dict[str, int | float]:
    """Read key grid fields from GAMMA dem_seg.par."""
    vals: dict[str, str] = {}
    for line in dem_par_file.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        vals[key.strip()] = value.strip().split()[0]

    required = ("width", "nlines", "corner_lat", "corner_lon", "post_lat", "post_lon")
    missing = [k for k in required if k not in vals]
    if missing:
        msg = f"Missing keys in {dem_par_file}: {missing}"
        raise ValueError(msg)

    return {
        "width": int(vals["width"]),
        "nlines": int(vals["nlines"]),
        "corner_lat": float(vals["corner_lat"]),
        "corner_lon": float(vals["corner_lon"]),
        "post_lat": float(vals["post_lat"]),
        "post_lon": float(vals["post_lon"]),
    }


def _extract_best_lookup_coordinates(
    *,
    lookup: np.ndarray,
    in_width: int,
    in_height: int,
    row_stride: int,
    col_stride: int,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Decode lookup coordinates robustly across common endian/component conventions."""
    lookup_be = lookup.view(np.dtype(">c8"))
    candidates = [
        ("native_real_imag", lookup.real, lookup.imag),
        ("native_imag_real", lookup.imag, lookup.real),
        ("big_real_imag", lookup_be.real, lookup_be.imag),
        ("big_imag_real", lookup_be.imag, lookup_be.real),
    ]

    best: tuple[np.ndarray, np.ndarray, str, float] | None = None
    for tag, x_raw, y_raw in candidates:
        x = np.full(x_raw.shape, np.nan, dtype=np.float64)
        y = np.full(y_raw.shape, np.nan, dtype=np.float64)
        finite_xy = np.isfinite(x_raw) & np.isfinite(y_raw)
        x[finite_xy] = x_raw[finite_xy].astype(np.float64) / col_stride
        y[finite_xy] = y_raw[finite_xy].astype(np.float64) / row_stride
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0)
            & (x <= in_width - 1)
            & (y >= 0)
            & (y <= in_height - 1)
        )
        score = float(np.count_nonzero(valid)) / valid.size
        if best is None or score > best[3]:
            best = (x, y, tag, score)

    assert best is not None
    return best


def _resample_from_lookup(
    *,
    in_arr: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    method: str,
    nodata: float | None,
) -> np.ndarray:
    """Resample input array onto output grid using lookup-table coordinates."""
    arr = np.asarray(in_arr)
    out_shape = x_coords.shape

    if method in {"near", "nearest", "nearest-neighbor"}:
        xi = np.rint(x_coords).astype(np.int64)
        yi = np.rint(y_coords).astype(np.int64)
        valid = (
            np.isfinite(x_coords)
            & np.isfinite(y_coords)
            & (xi >= 0)
            & (xi < arr.shape[1])
            & (yi >= 0)
            & (yi < arr.shape[0])
        )
        out = np.full(
            out_shape, nodata if nodata is not None else np.nan, dtype=arr.dtype
        )
        out[valid] = arr[yi[valid], xi[valid]]
        return out

    if method not in {"bilinear", "linear", "cubic", "cubicspline", "lanczos"}:
        msg = f"Unsupported resampling method for GAMMA lookup geocode: {method}"
        raise ValueError(msg)

    x0 = np.floor(x_coords).astype(np.int64)
    y0 = np.floor(y_coords).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    in_bounds = (
        np.isfinite(x_coords)
        & np.isfinite(y_coords)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < arr.shape[1])
        & (y1 < arr.shape[0])
    )

    out = np.full(out_shape, nodata if nodata is not None else np.nan, dtype=np.float32)
    if not np.any(in_bounds):
        return out

    x0v = x0[in_bounds]
    y0v = y0[in_bounds]
    x1v = x1[in_bounds]
    y1v = y1[in_bounds]
    xv = x_coords[in_bounds]
    yv = y_coords[in_bounds]

    q11 = arr[y0v, x0v]
    q21 = arr[y0v, x1v]
    q12 = arr[y1v, x0v]
    q22 = arr[y1v, x1v]

    if nodata is None:
        corner_ok = (
            np.isfinite(q11) & np.isfinite(q21) & np.isfinite(q12) & np.isfinite(q22)
        )
    else:
        corner_ok = (
            np.isfinite(q11)
            & np.isfinite(q21)
            & np.isfinite(q12)
            & np.isfinite(q22)
            & (q11 != nodata)
            & (q21 != nodata)
            & (q12 != nodata)
            & (q22 != nodata)
        )

    if not np.any(corner_ok):
        return out

    dx = xv - x0v
    dy = yv - y0v
    interp = (
        q11 * (1 - dx) * (1 - dy)
        + q21 * dx * (1 - dy)
        + q12 * (1 - dx) * dy
        + q22 * dx * dy
    )

    out_valid = out[in_bounds]
    out_valid[corner_ok] = interp[corner_ok].astype(np.float32)
    out[in_bounds] = out_valid
    return out


def _find_gamma_lookup_files(geometry_dir: Path) -> tuple[Path, Path]:
    """Find GAMMA lookup + dem_seg.par files inside a geometry directory."""
    lookup_candidates = [
        geometry_dir / "lookup_fine",
        geometry_dir / "lookup",
        geometry_dir / "*.lt_fine",
        geometry_dir / "*.lt",
    ]
    expanded: list[Path] = []
    for c in lookup_candidates:
        if "*" in str(c):
            expanded.extend(sorted(geometry_dir.glob(c.name)))
        elif c.exists():
            expanded.append(c)

    dem_par = geometry_dir / "dem_seg.par"
    if not expanded or not dem_par.exists():
        msg = (
            f"No GAMMA lookup/dem_seg.par found in {geometry_dir}. "
            "Expected lookup_fine/lookup/*.lt* and dem_seg.par"
        )
        raise FileNotFoundError(msg)
    return expanded[0], dem_par


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


def find_rasters_to_geocode(
    dolphin_work_dir: Path,
    *,
    include_interferograms: bool = False,
    include_unwrapped: bool = True,
    include_auxiliary: bool = False,
) -> list[Path]:
    """Find dolphin output rasters to geocode from the standard directory layout.

    Parameters
    ----------
    dolphin_work_dir : Path
        Path to dolphin work directory containing timeseries/, unwrapped/, etc.
    include_interferograms : bool
        Include wrapped interferograms, similarity, temporal coherence, and
        multilooked coherence.
    include_unwrapped : bool
        Include unwrapped interferograms.
    include_auxiliary : bool
        Include auxiliary products (CRLB, amplitude dispersion).

    Returns
    -------
    list[Path]
        Sorted list of raster paths found.

    """
    rasters: list[Path] = []

    # Time series outputs (displacement + residuals + velocity)
    ts_dir = dolphin_work_dir / "timeseries"
    if ts_dir.exists():
        rasters.extend(sorted(ts_dir.glob("[0-9]*_[0-9]*.tif")))
        rasters.extend(sorted(ts_dir.glob("residuals_[0-9]*_[0-9]*.tif")))
        velocity = ts_dir / "velocity.tif"
        if velocity.exists():
            rasters.append(velocity)

    if include_unwrapped:
        unw_dir = dolphin_work_dir / "unwrapped"
        if unw_dir.exists():
            rasters.extend(sorted(unw_dir.glob("*.unw.tif")))

    if include_interferograms:
        ifg_dir = dolphin_work_dir / "interferograms"
        if ifg_dir.exists():
            rasters.extend(sorted(ifg_dir.glob("*.int.tif")))
            rasters.extend(sorted(ifg_dir.glob("*.int.cor.tif")))
            rasters.extend(sorted(ifg_dir.glob("similarity_*.tif")))
            rasters.extend(sorted(ifg_dir.glob("temporal_coherence*.tif")))
            rasters.extend(sorted(ifg_dir.glob("multilooked_coherence*.tif")))

    if include_auxiliary:
        ifg_dir = dolphin_work_dir / "interferograms"
        if ifg_dir.exists():
            rasters.extend(sorted(ifg_dir.glob("crlb_*.tif")))
            amp_disp = ifg_dir / "amp_dispersion_looked.tif"
            if amp_disp.exists():
                rasters.append(amp_disp)

    return rasters


def _to_geocoded_path(
    in_file: Path,
    *,
    work_dir: Path,
    geocoded_dir: Path,
) -> Path:
    """Map an input file path to its geocoded output, mirroring the directory tree."""
    in_path = in_file.resolve()
    wd = work_dir.resolve()
    try:
        rel = in_path.relative_to(wd)
        return geocoded_dir / rel
    except ValueError:
        return geocoded_dir / in_path.name


# (lat_name, lon_name) pairs in priority order
_GEOLOCATION_PATTERNS: list[tuple[str, str]] = [
    ("y.tif", "x.tif"),  # ISCE3 / dolphin
    ("lat.rdr", "lon.rdr"),  # ISCE2 topsStack & stripmapStack
    ("lat.rdr.full", "lon.rdr.full"),  # ISCE2 full-res variants
    ("lat.rdc", "lon.rdc"),  # GAMMA common naming
    ("lat.rdc.full", "lon.rdc.full"),  # GAMMA full-res variants
    ("latitude.rdc", "longitude.rdc"),  # GAMMA alternate naming
    ("lat.tif", "lon.tif"),  # GeoTIFF exports from external tooling
    ("latitude.tif", "longitude.tif"),  # GeoTIFF alternate naming
    ("lat.vrt", "lon.vrt"),  # Raw binary + VRT sidecar naming
    ("latitude.vrt", "longitude.vrt"),
]


def _find_lat_lon_files(geometry_dir: Path) -> tuple[Path, Path]:
    """Resolve latitude and longitude files from a geometry directory.

    Searches for ISCE3-style (y.tif/x.tif) and ISCE2-style (lat.rdr/lon.rdr)
    naming conventions.

    Parameters
    ----------
    geometry_dir : Path
        Directory containing geolocation rasters.

    Returns
    -------
    tuple[Path, Path]
        ``(lat_file, lon_file)`` paths.

    Raises
    ------
    FileNotFoundError
        If no recognized lat/lon pair is found.

    """
    for lat_name, lon_name in _GEOLOCATION_PATTERNS:
        lat_candidate = geometry_dir / lat_name
        lon_candidate = geometry_dir / lon_name
        if lat_candidate.exists() and lon_candidate.exists():
            return lat_candidate, lon_candidate

    tried = ", ".join(f"{lat}/{lon}" for lat, lon in _GEOLOCATION_PATTERNS)
    msg = f"No lat/lon files found in {geometry_dir}. Tried: {tried}"
    raise FileNotFoundError(msg)


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


def _geocode_one_gamma(
    in_out: tuple[Path, Path],
    *,
    geometry_dir: Path,
    lookup_file: Path,
    dem_par_file: Path,
    strides: tuple[int, int] | None,
    resampling_method: str,
    creation_options: Sequence[str],
) -> Path:
    """Geocode one file using GAMMA lookup table."""
    in_path, out_path = in_out
    geocode_with_gamma_lookup_table(
        input_file=in_path,
        geometry_dir=geometry_dir,
        lookup_file=lookup_file,
        dem_par_file=dem_par_file,
        output_file=out_path,
        strides=strides,
        resampling_method=resampling_method,
        creation_options=creation_options,
    )
    return out_path


def run(
    input_files: Annotated[
        list[Path] | None,
        tyro.conf.arg(
            aliases=["-i", "--input"],
            help="Input file(s) to geocode. Can be specified multiple times.",
        ),
    ] = None,
    dolphin_dir: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-d"],
            help=(
                "Dolphin work directory to bulk-geocode. Auto-discovers rasters"
                " from timeseries/, unwrapped/, interferograms/ subdirectories."
                " Output mirrors the directory structure under <dolphin-dir>/geocoded/."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-o"],
            help=(
                "Output path. For single input, this is the output file path. "
                "For multiple inputs or --dolphin-dir, this should be a directory."
                " Default for --dolphin-dir: <dolphin-dir>/geocoded/."
            ),
        ),
    ] = None,
    geometry_dir: Annotated[
        Path | None,
        tyro.conf.arg(
            aliases=["-g", "--geometry"],
            help=(
                "Geometry directory containing lat/lon files. "
                "Auto-detects ISCE3 (y.tif/x.tif) and ISCE2 (lat.rdr/lon.rdr)."
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
                " In bulk mode (-d), if omitted and <dolphin-dir>/dolphin_config.yaml"
                " exists, it is used automatically."
            ),
        ),
    ] = None,
    include_interferograms: bool = False,
    include_unwrapped: bool = True,
    include_auxiliary: bool = False,
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

    Provide either ``-i`` for specific files or ``-d`` to bulk-geocode
    all outputs in a dolphin work directory.

    Examples
    --------
    Bulk geocode a dolphin work directory:

        dolphin geocode -d ./dolphin_output -g geometry/

    Equivalent explicit form (recommended when running from another directory):

        dolphin geocode -d ./dolphin_output -g geometry/ -c ./dolphin_output/dolphin_config.yaml

    Single file with geometry directory:

        dolphin geocode -i interferogram.tif -g geometry/

    Multiple files to output directory:

        dolphin geocode -i ifg.tif -i coherence.tif -g geometry/ -o geocoded/

    With mask and strides from dolphin config (for multilooked outputs):

        dolphin geocode -i ifg.tif -g geometry/ --mask mask.tif -c dolphin_config.yaml

    Output to UTM with 30m spacing:

        dolphin geocode -i ifg.tif -g geometry/ --srs 32610 -s 30

    """
    from dolphin._log import setup_logging

    setup_logging(logger_name="dolphin")

    if not input_files and dolphin_dir is None:
        msg = "Must provide either --input/-i files or --dolphin-dir/-d"
        raise ValueError(msg)

    geometry_mode = "latlon"
    gamma_lookup_file: Path | None = None
    gamma_dem_par_file: Path | None = None

    # Resolve geometry source
    if geometry_dir is not None:
        geometry_dir = Path(geometry_dir)
        try:
            lat_file, lon_file = _find_lat_lon_files(geometry_dir)
            logger.info("Using lat=%s, lon=%s", lat_file.name, lon_file.name)
        except FileNotFoundError:
            gamma_lookup_file, gamma_dem_par_file = _find_gamma_lookup_files(
                geometry_dir
            )
            geometry_mode = "gamma_lookup"
            logger.info(
                "Using GAMMA lookup=%s, dem_par=%s",
                gamma_lookup_file.name,
                gamma_dem_par_file.name,
            )

    if geometry_mode == "latlon":
        if lat_file is None or lon_file is None:
            msg = (
                "Must provide either --geometry, or both --lat and --lon. For GAMMA"
                " lookup geocoding, pass --geometry containing lookup_fine/lookup and"
                " dem_seg.par."
            )
            raise ValueError(msg)

        lat_file = Path(lat_file)
        lon_file = Path(lon_file)
        assert lat_file.exists(), f"Lat file not found: {lat_file}"
        assert lon_file.exists(), f"Lon file not found: {lon_file}"

    # Read strides from dolphin config
    if config is None and dolphin_dir is not None:
        auto_cfg = Path(dolphin_dir) / "dolphin_config.yaml"
        if auto_cfg.exists():
            config = auto_cfg
            logger.info("Auto-using config from dolphin_dir: %s", config)

    parsed_strides: tuple[int, int] | None = None
    if config is not None:
        from dolphin.workflows.config import DisplacementWorkflow

        cfg = DisplacementWorkflow.from_yaml(config)
        sy = cfg.output_options.strides.y
        sx = cfg.output_options.strides.x
        if sy != 1 or sx != 1:
            parsed_strides = (sy, sx)
            logger.info("Using strides from config: (%d, %d)", sy, sx)

    if geometry_mode == "gamma_lookup" and parsed_strides is None:
        logger.warning(
            "No strides configured for GAMMA lookup geocoding. If inputs are"
            " multilooked, pass -c dolphin_config.yaml (or place it under"
            " --dolphin-dir)."
        )

    # Parse spacing
    parsed_spacing: tuple[float, float] | None = None
    if spacing is not None:
        parsed_spacing = (spacing, spacing)

    # Discover or use provided input files
    if dolphin_dir is not None:
        dolphin_dir = Path(dolphin_dir).resolve()
        rasters = find_rasters_to_geocode(
            dolphin_dir,
            include_interferograms=include_interferograms,
            include_unwrapped=include_unwrapped,
            include_auxiliary=include_auxiliary,
        )
        if input_files:
            rasters.extend(input_files)
        logger.info("Found %d rasters to geocode", len(rasters))

        # Default output is <dolphin_dir>/geocoded/
        if output is None:
            output = dolphin_dir / "geocoded"
        output_dir = Path(output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build (input, output) pairs mirroring directory structure
        io_pairs: list[tuple[Path, Path]] = []
        skipped: list[Path] = []
        for raster in rasters:
            out_path = _to_geocoded_path(
                raster, work_dir=dolphin_dir, geocoded_dir=output_dir
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.stat().st_mtime >= raster.stat().st_mtime:
                logger.debug("Skipping %s (already exists)", out_path.name)
                skipped.append(out_path)
            else:
                io_pairs.append((raster, out_path))
    else:
        assert input_files is not None
        multiple_inputs = len(input_files) > 1
        output_is_dir = output is not None and (output.is_dir() or multiple_inputs)

        if multiple_inputs and output is not None:
            output.mkdir(parents=True, exist_ok=True)

        io_pairs = []
        skipped = []
        for in_file in input_files:
            in_path = Path(in_file)
            if output is None:
                out_path = in_path.with_suffix(f".geo{in_path.suffix}")
            elif output_is_dir:
                out_path = output / f"{in_path.stem}.geo{in_path.suffix}"
            else:
                out_path = output

            if (
                out_path.exists()
                and out_path.stat().st_mtime >= in_path.stat().st_mtime
            ):
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

    if geometry_mode == "gamma_lookup":
        if mask is not None:
            logger.warning("Mask is currently ignored in GAMMA lookup geocoding mode")
        worker = partial(
            _geocode_one_gamma,
            geometry_dir=geometry_dir,
            lookup_file=gamma_lookup_file,
            dem_par_file=gamma_dem_par_file,
            strides=parsed_strides,
            resampling_method=resampling_method,
            creation_options=creation_options,
        )
    else:
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
