"""Split a single frame into synthetic azimuth-block "bursts".

Used by :mod:`displacement` when the input does not match OPERA's burst
naming convention (e.g. NISAR GSLCs, where there is one frame-sized file
per acquisition rather than one file per Sentinel-1-style burst).

Each block reuses the full input file list and runs phase linking over an
expanded ``read_bounds`` rectangle (central rows + a row-direction halo).
The halo gives phase linking, similarity, and SHP estimation enough
neighborhood context that the **central** rows aren't edge-degraded.
After phase linking completes, each block's stitching-bound output
rasters are cropped to ``central_bounds`` (no halo) so adjacent blocks
have disjoint extents. The block-stitcher then mosaics the per-block
outputs without overlap, matching how OPERA bursts are stitched.

NOTE: each block here still uses a full-frame VRTStack and rasterizes a
full-frame bounds mask. ``EagerLoader`` skips fully-masked tiles, so I/O
stays bounded — but per-block RAM and VRT-init time scale with the
*full* frame, not the block, and N parallel blocks open N copies of the
same file handles concurrently. A leaner alternative is a per-block
windowed sub-VRT; deferred until benchmarks justify the refactor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from dolphin import io
from dolphin._types import Bbox, Filename

if TYPE_CHECKING:
    from .config import DisplacementWorkflow

__all__ = ["BlockBounds", "split_frame_into_blocks"]

logger = logging.getLogger(__name__)

# ``displacement.run`` hardcodes ``stitching_bursts.run(corr_window_size=(11, 11))``.
# The halo must cover the y-radius of that window so post-stitch correlation
# estimation has clean neighborhoods at block boundaries.
_STITCHER_CORR_WINDOW_Y = 11

# A few input rows of safety on top of the worst-case crop, to avoid cropping
# right at the edge of where phase linking actually has valid context.
_DEFAULT_HALO_SAFETY = 5


@dataclass(frozen=True)
class BlockBounds:
    """Per-block bounds for an azimuth-split synthetic burst.

    Attributes
    ----------
    read_bounds
        Bbox covering central rows plus a halo of input rows on top and
        bottom. Passed to ``output_options.bounds`` so the per-block
        phase-linking run sees the halo as valid input context.
    central_bounds
        Bbox covering only the central rows. After phase linking,
        per-block stitching-bound outputs are cropped to this so
        adjacent blocks have disjoint extents.
    epsg
        EPSG code of both bounds. Same as the input frame's CRS.

    """

    read_bounds: Bbox
    central_bounds: Bbox
    epsg: int


def _is_nisar_h5(filename: Filename) -> bool:
    """Detect NISAR raw HDF5 by filename prefix (matches `format_nc_filename`)."""
    s = str(filename)
    return s.endswith(".h5") and Path(s).name.upper().startswith("NISAR_")


def _gdal_path_for(cfg: DisplacementWorkflow) -> str:
    """Build the GDAL-compatible URI for the first input file in `cfg`.

    Uses ``dolphin.io.format_nc_filename`` so NISAR raw HDF5 (HDF5: driver),
    OPERA CSLCs and other CF-compliant HDF5s (NETCDF: driver), and plain
    GDAL-readable rasters all resolve correctly.
    """
    return io.format_nc_filename(cfg.cslc_file_list[0], cfg.input_options.subdataset)


def _read_nisar_grid_metadata(
    filename: Filename, subdataset: str
) -> tuple[int, int, tuple[float, ...], int]:
    """Read NISAR grid metadata via h5py.

    NISAR's GSLC files store the projection in a sibling ``projection``
    dataset and grid coordinates in ``xCoordinates`` / ``yCoordinates``
    (cell centers) inside the same group as the polarization dataset.
    GDAL's HDF5 driver doesn't expose any of this — it returns an identity
    geotransform — so the only reliable path is to read it directly.

    Returns ``(nx, ny, geotransform, epsg)``.
    """
    import h5py

    grid_path = str(PurePosixPath(subdataset).parent)
    with h5py.File(str(filename), "r") as f:
        # Honor the CF-style ``grid_mapping`` attribute if present; fall back
        # to the conventional ``projection`` name.
        dset = f[subdataset]
        proj_name = dset.attrs.get("grid_mapping", "projection")
        if isinstance(proj_name, bytes):
            proj_name = proj_name.decode()
        proj_raw = f[f"{grid_path}/{proj_name}"][()]
        epsg = int(proj_raw.decode()) if isinstance(proj_raw, bytes) else int(proj_raw)
        x_coords = f[f"{grid_path}/xCoordinates"][:]
        y_coords = f[f"{grid_path}/yCoordinates"][:]
        dx = float(f[f"{grid_path}/xCoordinateSpacing"][()])
        dy = float(f[f"{grid_path}/yCoordinateSpacing"][()])
    # NISAR coords are cell centers; geotransform anchors at the upper-left
    # cell *edge*, so back off half a pixel. dy is negative in standard
    # north-up NISAR grids, so subtracting dy/2 raises ymax above the first
    # row's center.
    gt = (
        float(x_coords[0]) - dx / 2.0,
        dx,
        0.0,
        float(y_coords[0]) - dy / 2.0,
        0.0,
        dy,
    )
    nx, ny = len(x_coords), len(y_coords)
    return nx, ny, gt, epsg


def _read_grid_metadata(
    cfg: DisplacementWorkflow,
) -> tuple[int, int, tuple[float, ...], int]:
    """Return ``(nx, ny, geotransform, epsg)`` for the first input file.

    NISAR raw HDF5: bypass GDAL and read from the grid group via h5py.
    Everything else (.tif, OPERA CSLC .h5, .nc): route through dolphin.io
    GDAL helpers and ``format_nc_filename``.
    """
    first = cfg.cslc_file_list[0]
    if _is_nisar_h5(first):
        # ``input_options.subdataset`` is required by the pydantic validator
        # whenever any input is an .h5/.nc, so this assert is for mypy.
        subdataset = cfg.input_options.subdataset
        assert subdataset is not None
        return _read_nisar_grid_metadata(first, subdataset)

    gdal_path = _gdal_path_for(cfg)
    try:
        crs = io.get_raster_crs(gdal_path)
        epsg = crs.to_epsg()
    except Exception as e:
        msg = (
            f"Could not read a usable EPSG from {gdal_path}; azimuth-block"
            " splitting needs a geocoded input. Pass num_blocks=1 to skip,"
            " or geocode the input frames first."
        )
        raise RuntimeError(msg) from e
    if epsg is None:
        msg = (
            f"Input {gdal_path} has a projection but no EPSG authority code;"
            " azimuth-block splitting needs an EPSG. Reproject the input first."
        )
        raise RuntimeError(msg)
    nx, ny = io.get_raster_xysize(gdal_path)
    gt = tuple(io.get_raster_gt(gdal_path))
    return nx, ny, gt, epsg


def _min_halo_rows(cfg: DisplacementWorkflow) -> int:
    """Minimum halo (input rows) for a safe block edge.

    Three windows can crop the edge of phase-linking output. Two of them
    are sized in strided/output coordinates, so they get multiplied by
    ``stride_y`` to come back to input rows:

    - ``half_window_y``                            (input rows)
    - ``similarity_search_radius * stride_y``      (input rows)
    - ``(corr_window_y // 2) * stride_y``          (input rows)

    A small safety margin keeps the central output away from the actual
    cover-zone edge.
    """
    half_window_y = cfg.phase_linking.half_window.y
    # similarity_search_radius lives on PhaseLinkingOptions on some branches
    # but not on upstream main; fall back to dolphin's default of 7.
    sim_radius = getattr(cfg.phase_linking, "similarity_search_radius", 7)
    stride_y = cfg.output_options.strides.y
    return (
        max(
            half_window_y,
            sim_radius * stride_y,
            (_STITCHER_CORR_WINDOW_Y // 2) * stride_y,
        )
        + _DEFAULT_HALO_SAFETY
    )


def split_frame_into_blocks(
    cfg: DisplacementWorkflow,
    *,
    num_blocks: int,
    halo_rows: int | None = None,
) -> dict[str, BlockBounds]:
    """Compute per-block read/central bounds for NISAR-style azimuth splitting.

    The frame is split in the azimuth (row) direction into ``num_blocks``
    non-overlapping **central** regions. Each block is also given a halo
    of ``halo_rows`` input rows on top and bottom; the halo is included
    in ``read_bounds`` (so phase linking has clean neighborhood context
    at the central edge) but excluded from ``central_bounds`` (so the
    block's stitched extent is disjoint from neighbors').

    Parameters
    ----------
    cfg
        Displacement workflow config. Reads the first input file's
        georeference, the phase-linking half-window, the similarity
        search radius (if present), and the stride.
    num_blocks
        Number of azimuth blocks (validated upstream by pydantic).
        When 1, returns a single block with no halo.
    halo_rows
        Halo on each side of a block, in input rows. ``None`` uses
        :func:`_min_halo_rows`.

    Returns
    -------
    dict[str, BlockBounds]
        Maps ``"block_00"``, ``"block_01"``, ... to per-block bounds.

    Raises
    ------
    ValueError
        If the central region would be smaller than ``2 * halo`` rows
        (caller should reduce ``num_blocks`` or pass ``halo_rows``).
    RuntimeError
        If the first input file has no usable projection.

    """
    halo = halo_rows if halo_rows is not None else _min_halo_rows(cfg)

    nx, ny, gt, epsg = _read_grid_metadata(cfg)
    xmin, ymax = gt[0], gt[3]
    px_x, px_y = gt[1], abs(gt[5])
    xmax = xmin + nx * px_x

    if num_blocks == 1:
        full = Bbox(float(xmin), float(ymax - ny * px_y), float(xmax), float(ymax))
        return {"block_00": BlockBounds(full, full, epsg)}

    rows_per = ny // num_blocks
    if rows_per <= 2 * halo:
        msg = (
            f"halo_rows={halo} too large for num_blocks={num_blocks} on a "
            f"{ny}-row frame: each block's central region would be "
            f"{rows_per} rows. Reduce num_blocks or pass halo_rows explicitly."
        )
        raise ValueError(msg)

    blocks: dict[str, BlockBounds] = {}
    for i in range(num_blocks):
        cs = i * rows_per
        ce = ny if i == num_blocks - 1 else (i + 1) * rows_per
        rs = max(0, cs - halo)
        re = min(ny, ce + halo)
        read_bbox = Bbox(
            float(xmin),
            float(ymax - re * px_y),
            float(xmax),
            float(ymax - rs * px_y),
        )
        central_bbox = Bbox(
            float(xmin),
            float(ymax - ce * px_y),
            float(xmax),
            float(ymax - cs * px_y),
        )
        block_id = f"block_{i:02d}"
        blocks[block_id] = BlockBounds(read_bbox, central_bbox, epsg)
        logger.debug(
            "%s: rows central=[%d, %d] read=[%d, %d] central_y=[%.0f, %.0f]",
            block_id,
            cs,
            ce,
            rs,
            re,
            central_bbox.bottom,
            central_bbox.top,
        )

    logger.info(
        "Split frame (%dx%d px, EPSG:%d) into %d azimuth blocks, halo=%d rows",
        nx,
        ny,
        epsg,
        num_blocks,
        halo,
    )
    return blocks


def crop_to_central(filename: Filename, central_bounds: Bbox) -> None:
    """Crop a raster in place to ``central_bounds``.

    Used post-phase-linking on each block's stitching-bound outputs so
    adjacent blocks have disjoint extents.

    Parameters
    ----------
    filename
        Path to the raster. Driver is preserved (e.g. GTiff, VRT).
    central_bounds
        Bbox in the raster's CRS.

    """
    from pathlib import Path

    from osgeo import gdal

    gdal.UseExceptions()
    src = Path(filename)
    # Use a sibling temp path that keeps the same extension so GDAL infers
    # the same driver. Insert ``.cropped`` before the extension.
    tmp = src.with_name(src.stem + ".cropped" + src.suffix)
    # projWin order: (ulx, uly, lrx, lry)
    proj_win = (
        central_bounds.left,
        central_bounds.top,
        central_bounds.right,
        central_bounds.bottom,
    )
    gdal.Translate(str(tmp), str(src), projWin=proj_win)
    src.unlink()
    tmp.rename(src)
