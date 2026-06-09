"""Tests for the NISAR azimuth-block splitter."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import h5py
import numpy as np
import pytest
from osgeo import gdal, osr

from dolphin._types import Bbox
from dolphin.workflows._block_split import (
    BlockBounds,
    _min_halo_rows,
    _read_grid_metadata,
    crop_to_central,
    split_frame_into_blocks,
    stitch_compressed_slcs,
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
    out = crop_to_central(target, bb.central_bounds)

    assert out == target  # TIF input → same path returned
    ds = gdal.Open(str(target))
    expected_ny = round((bb.central_bounds.top - bb.central_bounds.bottom) / 30)
    assert ds.RasterYSize == expected_ny
    assert ds.RasterXSize == 1000  # x extent unchanged
    # No stray .cropped sibling
    assert not (tmp_path / "to_crop.cropped.tif").exists()


def test_crop_to_central_materializes_vrt(dated_geotiff, tmp_path):
    """Cropping a .vrt materializes to a sibling .tif; original .vrt is removed.

    Regression test for ``RuntimeError: Recursion detected`` from gdal_merge:
    ``gdal.Translate`` on a ``.vrt`` input emits a VRT whose
    ``<SourceFilename>`` is the input path, so renaming it back over the
    source produces a self-referential VRT.
    """
    cfg = _make_cfg(dated_geotiff, tmp_path)
    blocks = split_frame_into_blocks(cfg, num_blocks=3)
    bb = blocks["block_01"]

    # Build a VRT that points at the source GeoTIFF (mirrors how dolphin's
    # virtual interferograms are structured).
    vrt = tmp_path / "to_crop.vrt"
    gdal.Translate(str(vrt), str(dated_geotiff), format="VRT")

    out = crop_to_central(vrt, bb.central_bounds)

    # Materialized to a sibling .tif at a new path
    assert out == tmp_path / "to_crop.tif"
    assert out.exists()
    assert not vrt.exists()  # original .vrt removed

    # And the output is openable as a non-recursive GTiff at the central extent
    ds = gdal.Open(str(out))
    assert ds.GetDriver().ShortName == "GTiff"
    expected_ny = round((bb.central_bounds.top - bb.central_bounds.bottom) / 30)
    assert ds.RasterYSize == expected_ny


def test_block_bounds_dataclass_immutable():
    bb = BlockBounds(
        read_bounds=Bbox(0.0, 0.0, 1.0, 1.0),
        central_bounds=Bbox(0.0, 0.0, 1.0, 1.0),
        epsg=4326,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bb.epsg = 32610  # type: ignore[misc]


def _make_nisar_h5(path: Path, *, nx: int = 100, ny: int = 200, epsg: int = 32637):
    """Build a minimal NISAR-shaped HDF5 with the projection + coord datasets.

    Mirrors the layout dolphin reads: ``/science/LSAR/GSLC/grids/frequencyA/{HH,
    projection, xCoordinates, yCoordinates, x/yCoordinateSpacing}``.
    """
    dx, dy = 10.0, -5.0
    x0, y0 = 500000.0, 1500000.0
    with h5py.File(path, "w") as f:
        g = f.create_group("/science/LSAR/GSLC/grids/frequencyA")
        hh = g.create_dataset("HH", shape=(ny, nx), dtype=np.complex64)
        hh.attrs["grid_mapping"] = np.bytes_(b"projection")
        g.create_dataset("projection", data=np.uint32(epsg))
        g.create_dataset(
            "xCoordinates", data=x0 + dx * np.arange(nx) + dx / 2.0, dtype="float64"
        )
        g.create_dataset(
            "yCoordinates", data=y0 + dy * np.arange(ny) + dy / 2.0, dtype="float64"
        )
        g.create_dataset("xCoordinateSpacing", data=dx)
        g.create_dataset("yCoordinateSpacing", data=dy)


def test_nisar_metadata_read_via_h5py(tmp_path):
    fn = tmp_path / "NISAR_L2_PR_GSLC_001_X_20250101_001.h5"
    _make_nisar_h5(fn, nx=100, ny=200, epsg=32637)
    cfg = DisplacementWorkflow(
        cslc_file_list=[fn],
        work_directory=tmp_path,
        input_options={"subdataset": "/science/LSAR/GSLC/grids/frequencyA/HH"},
    )
    nx, ny, gt, epsg = _read_grid_metadata(cfg)
    assert (nx, ny, epsg) == (100, 200, 32637)
    # cell-center -> UL-edge anchoring
    assert gt[0] == 500000.0
    assert gt[1] == 10.0
    assert gt[3] == 1500000.0  # y0 = y_coord_0 - dy/2 = (y0 + dy/2) - dy/2
    assert gt[5] == -5.0


def test_nisar_split_uses_h5py_path(tmp_path):
    fn = tmp_path / "NISAR_L2_PR_GSLC_001_X_20250101_001.h5"
    _make_nisar_h5(fn, nx=100, ny=6000, epsg=32637)
    cfg = DisplacementWorkflow(
        cslc_file_list=[fn],
        work_directory=tmp_path,
        input_options={"subdataset": "/science/LSAR/GSLC/grids/frequencyA/HH"},
        phase_linking={"half_window": {"y": DEFAULT_HALF_WINDOW_Y, "x": 11}},
        output_options={"strides": {"y": DEFAULT_STRIDE_Y, "x": 5}},
    )
    blocks = split_frame_into_blocks(cfg, num_blocks=3)
    assert len(blocks) == 3
    assert all(bb.epsg == 32637 for bb in blocks.values())
    # Centrals tile, reads overlap
    centrals = [blocks[f"block_{i:02d}"].central_bounds for i in range(3)]
    for prev, cur in itertools.pairwise(centrals):
        assert prev.bottom == cur.top


def test_vrtstack_patches_nisar_metadata(tmp_path):
    """``VRTStack`` must expose real CRS+GT for NISAR.

    Regression test for the all-zero ``bounds_mask`` failure that fired
    when the azimuth-block splitter set ``output_options.bounds`` on a
    NISAR run: GDAL's HDF5 driver returns identity geotransform + empty
    projection, so rasterizing a real-world polygon onto the VRT
    produced an empty mask and ``masking.MaskingError`` downstream.
    """
    from dolphin.io import VRTStack

    fn = tmp_path / "NISAR_L2_PR_GSLC_001_X_20250101_001.h5"
    _make_nisar_h5(fn, nx=50, ny=100, epsg=32637)
    vrt = VRTStack(
        [fn],
        outfile=tmp_path / "stack.vrt",
        subdataset="/science/LSAR/GSLC/grids/frequencyA/HH",
        sort_files=False,
        skip_size_check=True,
    )
    # Patched from h5py — NOT GDAL's identity / empty
    assert vrt.gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert vrt.gt[1] == 10.0  # dx
    assert vrt.gt[5] == -5.0  # dy
    assert "UTM zone 37N" in vrt.proj
    # And the .vrt on disk carries the same patched metadata
    ds = gdal.Open(str(tmp_path / "stack.vrt"))
    assert ds.GetGeoTransform() == vrt.gt
    assert "UTM zone 37N" in ds.GetProjection()


# ---------------------------------------------------------------------------
# stitch_compressed_slcs
# ---------------------------------------------------------------------------

_STITCH_PX = 30.0
_STITCH_ORIGIN_X = 500000.0
_STITCH_ORIGIN_Y = 4000000.0
_STITCH_EPSG = 32610


def _make_comp_slc(
    path: Path,
    bbox: Bbox,
    *,
    central_value: complex,
    halo_value: complex,
    halo_rows_top: int,
    halo_rows_bottom: int,
) -> None:
    """Write a 2-band CFloat32 compressed-SLC-like raster on the frame grid.

    Rows inside the halo margins are filled with ``halo_value`` (a sentinel)
    so a test can detect whether the (edge-degraded) halo wrongly won the
    mosaic overlap; central rows get ``central_value``.
    """
    nx = round((bbox.right - bbox.left) / _STITCH_PX)
    ny = round((bbox.top - bbox.bottom) / _STITCH_PX)
    arr = np.full((ny, nx), central_value, dtype=np.complex64)
    if halo_rows_top:
        arr[:halo_rows_top, :] = halo_value
    if halo_rows_bottom:
        arr[ny - halo_rows_bottom :, :] = halo_value

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), nx, ny, 2, gdal.GDT_CFloat32)
    ds.SetGeoTransform((bbox.left, _STITCH_PX, 0.0, bbox.top, 0.0, -_STITCH_PX))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(_STITCH_EPSG)
    ds.SetProjection(srs.ExportToWkt())
    for b in (1, 2):
        ds.GetRasterBand(b).WriteArray(arr)
    ds = None


def _bbox_for_rows(rs: int, re: int, nx: int) -> Bbox:
    """Bbox for input rows ``[rs, re)`` on the shared frame grid."""
    left = _STITCH_ORIGIN_X
    right = _STITCH_ORIGIN_X + nx * _STITCH_PX
    top = _STITCH_ORIGIN_Y - rs * _STITCH_PX
    bottom = _STITCH_ORIGIN_Y - re * _STITCH_PX
    return Bbox(left, bottom, right, top)


def test_stitch_compressed_slcs_mosaics_central_regions(tmp_path):
    """Per-block comp SLCs are cropped to central bounds and merged frame-sized.

    Two azimuth blocks share a 40-row frame split at row 20 with a 4-row halo.
    Each block's compressed SLC covers its read bounds (central + halo) with a
    sentinel value in the halo rows. After stitching, the halo must be gone and
    the frame must be tiled by each block's *central* data.
    """
    nx, ny = 8, 40
    split_row = 20
    halo = 4
    # block_00: central rows [0, 20), read rows [0, 24)  -> bottom halo only
    b00_central = _bbox_for_rows(0, split_row, nx)
    b00_read = _bbox_for_rows(0, split_row + halo, nx)
    # block_01: central rows [20, 40), read rows [16, 40) -> top halo only
    b01_central = _bbox_for_rows(split_row, ny, nx)
    b01_read = _bbox_for_rows(split_row - halo, ny, nx)

    block_bounds = {
        "block_00": BlockBounds(b00_read, b00_central, _STITCH_EPSG),
        "block_01": BlockBounds(b01_read, b01_central, _STITCH_EPSG),
    }

    name = "compressed_20200101_20200101_20200113.tif"
    halo_sentinel = 99 + 0j
    pl_dir = tmp_path / "phase_linking"
    (pl_dir / "block_00").mkdir(parents=True)
    (pl_dir / "block_01").mkdir(parents=True)
    f00 = pl_dir / "block_00" / name
    f01 = pl_dir / "block_01" / name
    _make_comp_slc(
        f00,
        b00_read,
        central_value=1 + 0j,
        halo_value=halo_sentinel,
        halo_rows_top=0,
        halo_rows_bottom=halo,
    )
    _make_comp_slc(
        f01,
        b01_read,
        central_value=2 + 0j,
        halo_value=halo_sentinel,
        halo_rows_top=halo,
        halo_rows_bottom=0,
    )

    out_dir = tmp_path / "compressed_slcs"
    new_dict = stitch_compressed_slcs(
        comp_slc_dict={"block_00": [f00], "block_01": [f01]},
        block_bounds=block_bounds,
        output_dir=out_dir,
        file_date_fmt="%Y%m%d",
    )

    # One frame-sized file under a single synthetic key, original name preserved.
    assert list(new_dict) == ["compressed_slc"]
    (stitched,) = new_dict["compressed_slc"]
    assert stitched.name == name
    assert stitched.parent == out_dir

    ds = gdal.Open(str(stitched))
    assert ds.RasterCount == 2
    assert (ds.RasterXSize, ds.RasterYSize) == (nx, ny)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Halo sentinel must be gone, and each half tiled by its block's central data.
    assert not np.any(arr == halo_sentinel)
    assert np.all(arr[:split_row, :] == 1)
    assert np.all(arr[split_row:, :] == 2)


def test_stitch_compressed_slcs_no_halo_passthrough(tmp_path):
    """A block whose read_bounds == central_bounds is merged uncropped."""
    nx, ny = 8, 20
    full = _bbox_for_rows(0, ny, nx)
    block_bounds = {"block_00": BlockBounds(full, full, _STITCH_EPSG)}
    name = "compressed_20200101_20200101_20200113.tif"
    f00 = tmp_path / "block_00" / name
    f00.parent.mkdir()
    _make_comp_slc(
        f00, full, central_value=1 + 0j, halo_value=0j, halo_rows_top=0,
        halo_rows_bottom=0,
    )

    out_dir = tmp_path / "compressed_slcs"
    new_dict = stitch_compressed_slcs(
        comp_slc_dict={"block_00": [f00]},
        block_bounds=block_bounds,
        output_dir=out_dir,
        file_date_fmt="%Y%m%d",
    )
    (stitched,) = new_dict["compressed_slc"]
    ds = gdal.Open(str(stitched))
    assert (ds.RasterXSize, ds.RasterYSize) == (nx, ny)
    ds = None
