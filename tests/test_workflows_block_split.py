"""Tests for the NISAR azimuth-block splitter."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import pytest
from osgeo import gdal, osr

from dolphin._types import Bbox
from dolphin.workflows._block_split import (
    BlockBounds,
    _min_halo_rows,
    crop_to_central,
    split_frame_into_blocks,
)
from dolphin.workflows.config import DisplacementWorkflow

gdal.UseExceptions()


# Defaults that exercise non-trivial halo math:
#   max(half_window_y=7, sim_radius=7 * stride_y=3, corr/2=5 * stride_y=3) + 5
DEFAULT_HALF_WINDOW_Y = 7
DEFAULT_STRIDE_Y = 3
EXPECTED_DEFAULT_HALO = 26


def _make_geotiff(
    path: Path,
    nx: int = 1000,
    ny: int = 6000,
    epsg: int | None = 32610,
    px: float = 30.0,
    origin_x: float = 500000.0,
    origin_y: float = 4000000.0,
) -> None:
    """Write a minimal complex32 GeoTIFF with the given size and CRS.

    Pass ``epsg=None`` to produce a file with no projection (negative test).
    """
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), nx, ny, 1, gdal.GDT_CFloat32)
    ds.SetGeoTransform((origin_x, px, 0.0, origin_y, 0.0, -px))
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
    ds = None


def _make_cfg(
    cslc_path: Path,
    work_dir: Path,
    *,
    half_window_y: int = DEFAULT_HALF_WINDOW_Y,
    stride_y: int = DEFAULT_STRIDE_Y,
) -> DisplacementWorkflow:
    return DisplacementWorkflow(
        cslc_file_list=[cslc_path],
        work_directory=work_dir,
        phase_linking={"half_window": {"y": half_window_y, "x": 11}},
        output_options={"strides": {"y": stride_y, "x": 5}},
    )


@pytest.fixture
def dated_geotiff(tmp_path: Path) -> Path:
    """A geocoded GeoTIFF with a parseable date in the filename."""
    fn = tmp_path / "20250101_slc.tif"
    _make_geotiff(fn)
    return fn


def test_min_halo_uses_max_of_window_sim_corr(tmp_path: Path):
    fn = tmp_path / "20250101_slc.tif"
    _make_geotiff(fn)
    cfg = _make_cfg(fn, tmp_path)
    # max(half_window_y=7, sim_radius=7 * stride_y=3, corr/2=5 * stride_y=3) + safety=5
    assert _min_halo_rows(cfg) == EXPECTED_DEFAULT_HALO


def test_num_blocks_1_returns_full_frame_no_halo(dated_geotiff, tmp_path):
    cfg = _make_cfg(dated_geotiff, tmp_path)
    blocks = split_frame_into_blocks(cfg, num_blocks=1)
    assert list(blocks.keys()) == ["block_00"]
    bb = blocks["block_00"]
    assert bb.read_bounds == bb.central_bounds
    assert bb.epsg == 32610
    assert bb.central_bounds == Bbox(
        500000.0, 4000000.0 - 6000 * 30.0, 500000.0 + 1000 * 30.0, 4000000.0
    )


def test_central_regions_tile_the_frame(dated_geotiff, tmp_path):
    cfg = _make_cfg(dated_geotiff, tmp_path)
    blocks = split_frame_into_blocks(cfg, num_blocks=3)
    centrals = [blocks[f"block_{i:02d}"].central_bounds for i in range(3)]
    # Adjacent centrals share an edge (no gap, no overlap)
    for prev, cur in itertools.pairwise(centrals):
        assert prev.bottom == cur.top
    # Top of first == top of frame; bottom of last == bottom of frame
    assert centrals[0].top == 4000000.0
    assert centrals[-1].bottom == 4000000.0 - 6000 * 30.0


def test_read_bounds_overlap_by_halo_on_interior_edges(dated_geotiff, tmp_path):
    cfg = _make_cfg(dated_geotiff, tmp_path)
    halo = _min_halo_rows(cfg)
    blocks = split_frame_into_blocks(cfg, num_blocks=3)
    centrals = [blocks[f"block_{i:02d}"].central_bounds for i in range(3)]
    reads = [blocks[f"block_{i:02d}"].read_bounds for i in range(3)]
    px_y = 30.0
    # First block: halo only on the bottom (top is at frame edge)
    assert reads[0].top == centrals[0].top
    assert reads[0].bottom == centrals[0].bottom - halo * px_y
    # Middle block: halo on both sides
    assert reads[1].top == centrals[1].top + halo * px_y
    assert reads[1].bottom == centrals[1].bottom - halo * px_y
    # Last block: halo only on the top
    assert reads[-1].top == centrals[-1].top + halo * px_y
    assert reads[-1].bottom == centrals[-1].bottom


def test_halo_too_large_raises(tmp_path):
    fn = tmp_path / "20250101_slc.tif"
    _make_geotiff(fn, ny=50)  # tiny frame
    cfg = _make_cfg(fn, tmp_path)
    with pytest.raises(ValueError, match="too large"):
        split_frame_into_blocks(cfg, num_blocks=10)


def test_missing_projection_raises_with_epsg_message(tmp_path):
    fn = tmp_path / "20250101_slc.tif"
    _make_geotiff(fn, epsg=None)
    cfg = _make_cfg(fn, tmp_path)
    with pytest.raises(RuntimeError, match="EPSG"):
        split_frame_into_blocks(cfg, num_blocks=2)


def test_crop_to_central_replaces_in_place(dated_geotiff, tmp_path):
    cfg = _make_cfg(dated_geotiff, tmp_path)
    blocks = split_frame_into_blocks(cfg, num_blocks=3)
    bb = blocks["block_01"]

    # Copy frame to a sibling, then crop in-place
    target = tmp_path / "to_crop.tif"
    gdal.Translate(str(target), str(dated_geotiff))
    crop_to_central(target, bb.central_bounds)

    ds = gdal.Open(str(target))
    expected_ny = round((bb.central_bounds.top - bb.central_bounds.bottom) / 30)
    assert ds.RasterYSize == expected_ny
    assert ds.RasterXSize == 1000  # x extent unchanged
    # Same path retained
    assert target.exists()
    # No stray .cropped sibling
    assert not (tmp_path / "to_crop.cropped.tif").exists()


def test_block_bounds_dataclass_immutable():
    bb = BlockBounds(
        read_bounds=Bbox(0.0, 0.0, 1.0, 1.0),
        central_bounds=Bbox(0.0, 0.0, 1.0, 1.0),
        epsg=4326,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bb.epsg = 32610  # type: ignore[misc]
