"""Split a single frame into synthetic azimuth-block "bursts".

Used by ``displacement.py`` when the input does not match OPERA's burst
naming convention (e.g. NISAR GSLCs, where there is one frame-sized file
per acquisition rather than one file per Sentinel-1-style burst).

Each block reuses the full input file list but applies its own
``output_options.bounds`` to restrict spatial processing. The block-stitcher
(``stitching_bursts.run``) then mosaics the per-block outputs the same way
it mosaics real OPERA bursts.

NOTE: each block here still uses a full-frame VRTStack and rasterizes a
full-frame bounds mask (`wrapped_phase._get_mask`). EagerLoader skips
fully-masked tiles, so I/O stays bounded — but per-block RAM and VRT-init
time scale with the *full* frame, not the block. With N blocks x 20 GSLCs,
that's also N copies of the same file handles open concurrently.

A leaner alternative is a per-block **windowed sub-VRT** (gdal.Translate
with srcWin=(0, y_off, nx, y_size), or exposing the SrcRect/DstRect hooks
already present in VRTStack's XML template via a new `pixel_window` kwarg).
That would:
  - cut per-block RAM (no full-frame mask, no full-frame VRT)
  - shave VRT init per block
  - eliminate boundary-tile read waste at block edges

The trade is a non-trivial refactor: VRTStack.shape / __getitem__ /
slice_dates all assume the full frame, and downstream `like_filename`
callers would now see a block-sized georef (which is what burst-stitching
already wants). Deferred — bounds-based blocking is benchmarked first.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dolphin._types import Bbox

if TYPE_CHECKING:
    from .config import DisplacementWorkflow

__all__ = ["split_frame_into_blocks"]

logger = logging.getLogger(__name__)

# ``displacement.run`` hardcodes ``stitching_bursts.run(corr_window_size=(11, 11))``
# (see displacement.py). The halo on each block must cover its radius so that
# stitching feathering uses non-edge-degraded data at block boundaries.
_STITCHER_CORR_WINDOW_Y = 11

# A few input rows of safety on top of the worst-case crop. Cheap insurance
# against off-by-one alignment with strides / block_shape.
_DEFAULT_HALO_SAFETY = 5


def _open_for_bounds(first_file: str, subdataset: str | None):
    """Open ``first_file`` (HDF5 subdataset if applicable) for georef inspection.

    Returns the GDAL dataset; caller is responsible for releasing it.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    path = f'NETCDF:"{first_file}":{subdataset}' if subdataset else str(first_file)
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    return ds


def _min_halo_rows(cfg: DisplacementWorkflow) -> int:
    """Compute the minimum halo (input rows) required for a safe block edge.

    The block edge gets cropped by the worst of three windows. Two are in
    strided/output coordinates, so they multiply by ``stride_y`` to come back
    to input rows:

        - half_window_y                                          (input rows)
        - similarity_search_radius * stride_y                    (input rows)
        - (corr_window_y // 2)     * stride_y                    (input rows)

    A small safety margin is added so per-block outputs aren't cropped right
    at the cover-zone edge.
    """
    half_window_y = cfg.phase_linking.half_window.y
    # similarity_search_radius lives on PhaseLinkingOptions on the similarity
    # branch but not on main; fall back to dolphin's default of 7 when absent.
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
) -> dict[str, tuple[Bbox, int]]:
    """Compute per-block (bounds, epsg) for synthetic NISAR-style bursts.

    The frame is split in the **azimuth** (row) direction into ``num_blocks``
    non-overlapping central regions; each region is padded on top and bottom
    by ``halo_rows`` to give phase-linking / similarity / stitching room to
    operate without edge artifacts.

    Parameters
    ----------
    cfg
        The full displacement workflow config. Used to read the first input
        file's georeference, the subdataset path, the phase-linking
        half-window, similarity search radius, and stride.
    num_blocks
        Number of azimuth blocks. Must be >= 1. When 1, the result is a
        single full-frame entry with no halo.
    halo_rows
        Halo on each side of a block, in **input rows**. When ``None``,
        defaults to ``max(half_window_y, similarity_search_radius * stride_y,
        (corr_window_y // 2) * stride_y) + 3``.

    Returns
    -------
    dict
        Maps block id (``"block_00"``, ``"block_01"``, ...) to a pair
        ``((xmin, ymin, xmax, ymax), epsg)`` suitable for assigning to a
        per-block ``output_options.bounds`` / ``output_options.bounds_epsg``.

    Raises
    ------
    ValueError
        If ``num_blocks < 1`` or ``cfg.cslc_file_list`` is empty.
    RuntimeError
        If the first CSLC cannot be opened with GDAL.

    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if not cfg.cslc_file_list:
        raise ValueError("cfg.cslc_file_list is empty; nothing to split")

    halo = halo_rows if halo_rows is not None else _min_halo_rows(cfg)
    if halo < 0:
        raise ValueError(f"halo_rows must be >= 0, got {halo}")

    from osgeo import osr

    first_file = str(cfg.cslc_file_list[0])
    subdataset = cfg.input_options.subdataset
    ds = _open_for_bounds(first_file, subdataset)
    try:
        gt = ds.GetGeoTransform()
        nx, ny = ds.RasterXSize, ds.RasterYSize
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        authority = srs.GetAttrValue("AUTHORITY", 1)
        if authority is None:
            raise RuntimeError(f"Could not read EPSG code from {first_file} projection")
        epsg = int(authority)
    finally:
        ds = None  # release GDAL handle

    xmin, ymax = gt[0], gt[3]
    px_x, px_y = gt[1], abs(gt[5])
    xmax = xmin + nx * px_x

    if num_blocks == 1:
        # Single block = full frame, no halo needed (no boundaries to feather).
        return {
            "block_00": (
                Bbox(float(xmin), float(ymax - ny * px_y), float(xmax), float(ymax)),
                epsg,
            )
        }

    rows_per = ny // num_blocks
    if rows_per <= 2 * halo:
        # Halo would exceed the central region — caller should reduce blocks
        # or halo. Refusing rather than silently producing degenerate blocks.
        raise ValueError(
            f"halo_rows={halo} too large for num_blocks={num_blocks} on a "
            f"{ny}-row frame: each block's central region would be "
            f"{rows_per} rows. Reduce num_blocks or pass halo_rows explicitly."
        )

    blocks: dict[str, tuple[Bbox, int]] = {}
    for i in range(num_blocks):
        cs = i * rows_per
        ce = ny if i == num_blocks - 1 else (i + 1) * rows_per
        rs = max(0, cs - halo)
        re = min(ny, ce + halo)
        block_ymax = ymax - rs * px_y
        block_ymin = ymax - re * px_y
        block_id = f"block_{i:02d}"
        blocks[block_id] = (
            Bbox(float(xmin), float(block_ymin), float(xmax), float(block_ymax)),
            epsg,
        )
        logger.debug(
            "%s: rows central=[%d, %d] read=[%d, %d] y=[%.0f, %.0f]",
            block_id,
            cs,
            ce,
            rs,
            re,
            block_ymin,
            block_ymax,
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
