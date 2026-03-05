"""Tests for the geocode module using synthetic geolocation arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from dolphin import io
from dolphin.geocode import (
    _find_lat_lon_files,
    find_rasters_to_geocode,
    geocode_with_gamma_lookup_table,
    geocode_with_geolocation_arrays,
    run,
)

gdal.UseExceptions()

# Synthetic scene centered near Mexico City
LAT_CENTER, LON_CENTER = 19.4, -99.1
NROWS, NCOLS = 60, 80


def _write_float32_raster(path: Path, data: np.ndarray) -> Path:
    """Write a 2D float32 array to a GeoTIFF (no geotransform/projection)."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None
    return path


@pytest.fixture()
def geometry_dir(tmp_path):
    """Create synthetic lat/lon geolocation rasters as a linear grid."""
    geo_dir = tmp_path / "geometry"
    geo_dir.mkdir()

    lat = np.linspace(LAT_CENTER + 0.3, LAT_CENTER - 0.3, NROWS).reshape(-1, 1)
    lat = np.broadcast_to(lat, (NROWS, NCOLS)).copy()
    lon = np.linspace(LON_CENTER - 0.4, LON_CENTER + 0.4, NCOLS).reshape(1, -1)
    lon = np.broadcast_to(lon, (NROWS, NCOLS)).copy()

    _write_float32_raster(geo_dir / "y.tif", lat.astype(np.float32))
    _write_float32_raster(geo_dir / "x.tif", lon.astype(np.float32))
    return geo_dir


@pytest.fixture()
def input_raster(tmp_path):
    """Create a synthetic input raster (a simple gradient)."""
    data = np.arange(NROWS * NCOLS, dtype=np.float32).reshape(NROWS, NCOLS)
    return _write_float32_raster(tmp_path / "input.tif", data)


@pytest.fixture()
def mask_raster(tmp_path, input_raster):
    """Create a mask that invalidates the left half of the image."""
    mask = np.ones((NROWS, NCOLS), dtype=np.uint8)
    mask[:, : NCOLS // 2] = 0
    path = tmp_path / "mask.tif"
    io.write_arr(arr=mask, output_name=path, like_filename=input_raster)
    return path


class TestGeocodeWithGeolocationArrays:
    def test_basic(self, tmp_path, geometry_dir, input_raster):
        out = tmp_path / "output.geo.tif"
        result = geocode_with_geolocation_arrays(
            input_file=input_raster,
            lat_file=geometry_dir / "y.tif",
            lon_file=geometry_dir / "x.tif",
            output_file=out,
        )
        assert result == out
        assert out.exists()

        ds = gdal.Open(str(out))
        gt = ds.GetGeoTransform()
        # Output should be placed in the right lon/lat neighborhood
        assert LON_CENTER - 1 < gt[0] < LON_CENTER + 1
        assert LAT_CENTER - 1 < gt[3] < LAT_CENTER + 1
        # Should have non-NaN data
        arr = ds.GetRasterBand(1).ReadAsArray()
        assert np.any(np.isfinite(arr))

    def test_default_output_name(self, geometry_dir, input_raster):
        result = geocode_with_geolocation_arrays(
            input_file=input_raster,
            lat_file=geometry_dir / "y.tif",
            lon_file=geometry_dir / "x.tif",
        )
        assert result.name == "input.geo.tif"
        assert result.exists()

    def test_with_mask(self, tmp_path, geometry_dir, input_raster, mask_raster):
        out = tmp_path / "masked.geo.tif"
        result = geocode_with_geolocation_arrays(
            input_file=input_raster,
            lat_file=geometry_dir / "y.tif",
            lon_file=geometry_dir / "x.tif",
            output_file=out,
            mask_file=mask_raster,
        )
        assert result.exists()
        ds = gdal.Open(str(out))
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        # Should have both valid and NaN pixels
        assert np.any(np.isfinite(arr))
        assert np.any(np.isnan(arr))

    def test_with_strides(self, tmp_path, geometry_dir):
        """Strided input: input is smaller than geolocation arrays."""
        stride_y, stride_x = 2, 2
        small_rows = NROWS // stride_y
        small_cols = NCOLS // stride_x
        small_data = np.arange(small_rows * small_cols, dtype=np.float32).reshape(
            small_rows, small_cols
        )
        small_input = _write_float32_raster(tmp_path / "strided_input.tif", small_data)

        out = tmp_path / "strided.geo.tif"
        result = geocode_with_geolocation_arrays(
            input_file=small_input,
            lat_file=geometry_dir / "y.tif",
            lon_file=geometry_dir / "x.tif",
            output_file=out,
            strides=(stride_y, stride_x),
        )
        assert result.exists()
        ds = gdal.Open(str(out))
        gt = ds.GetGeoTransform()
        assert LON_CENTER - 1 < gt[0] < LON_CENTER + 1
        arr = ds.GetRasterBand(1).ReadAsArray()
        assert np.any(np.isfinite(arr))

    def test_with_spacing(self, tmp_path, geometry_dir, input_raster):
        out = tmp_path / "spaced.geo.tif"
        geocode_with_geolocation_arrays(
            input_file=input_raster,
            lat_file=geometry_dir / "y.tif",
            lon_file=geometry_dir / "x.tif",
            output_file=out,
            spacing=0.01,
        )
        ds = gdal.Open(str(out))
        gt = ds.GetGeoTransform()
        assert abs(gt[1] - 0.01) < 1e-6
        assert abs(gt[5] + 0.01) < 1e-6


def _make_lat_lon_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic lat/lon arrays for the test scene."""
    lat = np.linspace(LAT_CENTER + 0.3, LAT_CENTER - 0.3, NROWS).reshape(-1, 1)
    lat = np.broadcast_to(lat, (NROWS, NCOLS)).copy().astype(np.float32)
    lon = np.linspace(LON_CENTER - 0.4, LON_CENTER + 0.4, NCOLS).reshape(1, -1)
    lon = np.broadcast_to(lon, (NROWS, NCOLS)).copy().astype(np.float32)
    return lat, lon


class TestFindLatLonFiles:
    def test_isce3_style(self, tmp_path):
        """Finds y.tif/x.tif (ISCE3 / dolphin convention)."""
        lat, lon = _make_lat_lon_arrays()
        _write_float32_raster(tmp_path / "y.tif", lat)
        _write_float32_raster(tmp_path / "x.tif", lon)
        lat_f, lon_f = _find_lat_lon_files(tmp_path)
        assert lat_f.name == "y.tif"
        assert lon_f.name == "x.tif"

    def test_isce2_style(self, tmp_path):
        """Finds lat.rdr/lon.rdr (ISCE2 topsStack / stripmapStack)."""
        lat, lon = _make_lat_lon_arrays()
        _write_float32_raster(tmp_path / "lat.rdr", lat)
        _write_float32_raster(tmp_path / "lon.rdr", lon)
        lat_f, lon_f = _find_lat_lon_files(tmp_path)
        assert lat_f.name == "lat.rdr"
        assert lon_f.name == "lon.rdr"

    def test_isce3_preferred_over_isce2(self, tmp_path):
        """When both exist, ISCE3 convention wins."""
        lat, lon = _make_lat_lon_arrays()
        _write_float32_raster(tmp_path / "y.tif", lat)
        _write_float32_raster(tmp_path / "x.tif", lon)
        _write_float32_raster(tmp_path / "lat.rdr", lat)
        _write_float32_raster(tmp_path / "lon.rdr", lon)
        lat_f, _lon_f = _find_lat_lon_files(tmp_path)
        assert lat_f.name == "y.tif"

    def test_gamma_style(self, tmp_path):
        """Finds lat.rdc/lon.rdc (common GAMMA convention)."""
        lat, lon = _make_lat_lon_arrays()
        _write_float32_raster(tmp_path / "lat.rdc", lat)
        _write_float32_raster(tmp_path / "lon.rdc", lon)
        lat_f, lon_f = _find_lat_lon_files(tmp_path)
        assert lat_f.name == "lat.rdc"
        assert lon_f.name == "lon.rdc"

    def test_error_no_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No lat/lon files found"):
            _find_lat_lon_files(tmp_path)

    def test_run_with_isce2_geometry(self, tmp_path):
        """End-to-end: run() auto-detects ISCE2 lat.rdr/lon.rdr."""
        geo_dir = tmp_path / "geom_reference"
        geo_dir.mkdir()
        lat, lon = _make_lat_lon_arrays()
        _write_float32_raster(geo_dir / "lat.rdr", lat)
        _write_float32_raster(geo_dir / "lon.rdr", lon)

        data = np.arange(NROWS * NCOLS, dtype=np.float32).reshape(NROWS, NCOLS)
        input_file = _write_float32_raster(tmp_path / "input.tif", data)

        out = tmp_path / "output.geo.tif"
        result = run(
            input_files=[input_file],
            geometry_dir=geo_dir,
            output=out,
        )
        assert len(result) == 1
        assert result[0].exists()
        ds = gdal.Open(str(out))
        gt = ds.GetGeoTransform()
        assert LON_CENTER - 1 < gt[0] < LON_CENTER + 1


class TestFindRastersToGeocode:
    @pytest.fixture()
    def dolphin_dir(self, tmp_path):
        """Create a mock dolphin directory structure with empty rasters."""
        d = tmp_path / "dolphin_work"
        ts_dir = d / "timeseries"
        ts_dir.mkdir(parents=True)
        unw_dir = d / "unwrapped"
        unw_dir.mkdir()
        ifg_dir = d / "interferograms"
        ifg_dir.mkdir()

        dummy = np.zeros((4, 4), dtype=np.float32)
        for name in [
            "timeseries/20220101_20220102.tif",
            "timeseries/20220102_20220103.tif",
            "timeseries/residuals_20220101_20220102.tif",
            "timeseries/velocity.tif",
            "unwrapped/20220101_20220102.unw.tif",
            "interferograms/20220101_20220102.int.tif",
            "interferograms/similarity_20220101.tif",
            "interferograms/temporal_coherence.tif",
            "interferograms/multilooked_coherence.tif",
            "interferograms/crlb_velocity.tif",
            "interferograms/amp_dispersion_looked.tif",
        ]:
            _write_float32_raster(d / name, dummy)

        return d

    def test_defaults(self, dolphin_dir):
        """By default, finds timeseries + unwrapped."""
        rasters = find_rasters_to_geocode(dolphin_dir)
        names = {r.name for r in rasters}
        assert "velocity.tif" in names
        assert "20220101_20220102.tif" in names
        assert "residuals_20220101_20220102.tif" in names
        assert "20220101_20220102.unw.tif" in names
        # Should NOT include interferograms or auxiliary by default
        assert "20220101_20220102.int.tif" not in names
        assert "crlb_velocity.tif" not in names

    def test_include_interferograms(self, dolphin_dir):
        rasters = find_rasters_to_geocode(dolphin_dir, include_interferograms=True)
        names = {r.name for r in rasters}
        assert "20220101_20220102.int.tif" in names
        assert "similarity_20220101.tif" in names
        assert "temporal_coherence.tif" in names
        assert "multilooked_coherence.tif" in names

    def test_include_auxiliary(self, dolphin_dir):
        rasters = find_rasters_to_geocode(dolphin_dir, include_auxiliary=True)
        names = {r.name for r in rasters}
        assert "crlb_velocity.tif" in names
        assert "amp_dispersion_looked.tif" in names

    def test_exclude_unwrapped(self, dolphin_dir):
        rasters = find_rasters_to_geocode(dolphin_dir, include_unwrapped=False)
        names = {r.name for r in rasters}
        assert "20220101_20220102.unw.tif" not in names


class TestRun:
    def test_single_file(self, tmp_path, geometry_dir, input_raster):
        out = tmp_path / "out.geo.tif"
        result = run(
            input_files=[input_raster],
            geometry_dir=geometry_dir,
            output=out,
        )
        assert len(result) == 1
        assert result[0].exists()

    def test_dolphin_dir(self, tmp_path, geometry_dir):
        """Test bulk geocode from a dolphin work directory."""
        # Build a minimal dolphin directory
        dolphin_dir = tmp_path / "dolphin_work"
        ts_dir = dolphin_dir / "timeseries"
        ts_dir.mkdir(parents=True)

        data = np.arange(NROWS * NCOLS, dtype=np.float32).reshape(NROWS, NCOLS)
        _write_float32_raster(ts_dir / "20220101_20220102.tif", data)
        _write_float32_raster(ts_dir / "velocity.tif", data * 0.1)

        result = run(
            dolphin_dir=dolphin_dir,
            geometry_dir=geometry_dir,
            include_unwrapped=False,
        )
        assert len(result) == 2
        geocoded_dir = dolphin_dir / "geocoded"
        assert geocoded_dir.exists()
        # Directory structure should be mirrored
        assert (geocoded_dir / "timeseries" / "20220101_20220102.tif").exists()
        assert (geocoded_dir / "timeseries" / "velocity.tif").exists()

    def test_skips_existing(self, tmp_path, geometry_dir, input_raster):
        """Running twice should skip on the second call."""
        out = tmp_path / "out.geo.tif"
        run(
            input_files=[input_raster],
            geometry_dir=geometry_dir,
            output=out,
        )
        # Second run should skip
        result = run(
            input_files=[input_raster],
            geometry_dir=geometry_dir,
            output=out,
        )
        assert len(result) == 1

    def test_error_no_inputs(self, geometry_dir):
        with pytest.raises(ValueError, match="Must provide"):
            run(geometry_dir=geometry_dir)


class TestGammaLookupGeocode:
    def _write_dem_par(self, path: Path, width: int, nlines: int) -> None:
        path.write_text(
            "\n".join(
                [
                    "Gamma DIFF&GEO DEM/MAP parameter file",
                    f"width:                {width}",
                    f"nlines:               {nlines}",
                    "corner_lat:     40.0  decimal degrees",
                    "corner_lon:    115.0  decimal degrees",
                    "post_lat: -1.0e-4  decimal degrees",
                    "post_lon:  1.0e-4  decimal degrees",
                ]
            ),
            encoding="utf-8",
        )

    def test_gamma_lookup_with_strides(self, tmp_path):
        """A 1x1 lookup table should geocode looked data after stride scaling."""
        in_rows, in_cols = 3, 4
        arr = np.arange(in_rows * in_cols, dtype=np.float32).reshape(in_rows, in_cols)
        input_file = _write_float32_raster(tmp_path / "velocity.tif", arr)

        # Build a synthetic 1x1 lookup in full-res coordinates for strides=(2, 2).
        stride_y, stride_x = 2, 2
        yy, xx = np.meshgrid(np.arange(in_rows), np.arange(in_cols), indexing="ij")
        lookup = (xx * stride_x).astype(np.float32) + 1j * (
            yy * stride_y
        ).astype(np.float32)

        geometry_dir = tmp_path / "DEM_prep"
        geometry_dir.mkdir()
        lookup_file = geometry_dir / "lookup_fine"
        lookup.astype(np.complex64).tofile(lookup_file)
        dem_par = geometry_dir / "dem_seg.par"
        self._write_dem_par(dem_par, width=in_cols, nlines=in_rows)

        out = tmp_path / "velocity.geo.tif"
        geocode_with_gamma_lookup_table(
            input_file=input_file,
            lookup_file=lookup_file,
            dem_par_file=dem_par,
            output_file=out,
            strides=(stride_y, stride_x),
            resampling_method="near",
        )

        ds = gdal.Open(str(out))
        got = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        np.testing.assert_allclose(got, arr)

    def test_run_auto_detect_gamma_lookup(self, tmp_path):
        in_rows, in_cols = 3, 4
        arr = np.arange(in_rows * in_cols, dtype=np.float32).reshape(in_rows, in_cols)
        input_file = _write_float32_raster(tmp_path / "velocity.tif", arr)

        geometry_dir = tmp_path / "DEM_prep"
        geometry_dir.mkdir()
        yy, xx = np.meshgrid(np.arange(in_rows), np.arange(in_cols), indexing="ij")
        lookup = xx.astype(np.float32) + 1j * yy.astype(np.float32)
        (geometry_dir / "lookup_fine").write_bytes(lookup.astype(np.complex64).tobytes())
        self._write_dem_par(geometry_dir / "dem_seg.par", width=in_cols, nlines=in_rows)

        out = tmp_path / "velocity.geo.tif"
        result = run(
            input_files=[input_file],
            geometry_dir=geometry_dir,
            output=out,
            resampling_method="near",
        )
        assert len(result) == 1
        assert result[0].exists()
