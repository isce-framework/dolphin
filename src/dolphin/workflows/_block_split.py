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
from pathlib import Path
from typing import TYPE_CHECKING

from dolphin import io
from dolphin._types import Bbox, Filename

if TYPE_CHECKING:
    from .config import DisplacementWorkflow

__all__ = ["BlockBounds", "split_frame_into_blocks", "stitch_compressed_slcs"]

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


def _read_grid_metadata(
    cfg: DisplacementWorkflow,
) -> tuple[int, int, tuple[float, ...], int]:
    """Return ``(nx, ny, geotransform, epsg)`` for the first input file.

    NISAR raw HDF5 reads via h5py because GDAL's HDF5 driver doesn't
    expose the CF grid_mapping; everything else routes through the
    ``dolphin.io`` GDAL helpers.
    """
    first = cfg.cslc_file_list[0]
    if io._core._is_nisar_h5(first):
        # ``input_options.subdataset`` is required by the pydantic validator
        # for any .h5/.nc; assert is for mypy.
        subdataset = cfg.input_options.subdataset
        assert subdataset is not None
        nx, ny, gt, epsg, _ = io._core.read_nisar_grid_metadata(first, subdataset)
        return nx, ny, gt, epsg

    gdal_path = io.format_nc_filename(first, cfg.input_options.subdataset)
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


def crop_to_central(filename: Filename, central_bounds: Bbox) -> Path:
    """Crop a raster to ``central_bounds`` and return the path of the result.

    Used post-phase-linking on each block's stitching-bound outputs so
    adjacent blocks have disjoint extents.

    Always materializes to GTiff. For ``.tif`` inputs the result replaces
    the original at the same path; for ``.vrt`` inputs the result is a
    sibling ``.tif`` and the original ``.vrt`` is removed. Materializing
    is necessary because ``gdal.Translate(out.vrt, src.vrt, projWin=...)``
    writes a VRT whose ``<SourceFilename>`` is the input path — renaming
    that output back over the source produces a self-referential VRT that
    gdal_merge later rejects with "Recursion detected".

    Parameters
    ----------
    filename
        Path to the raster (``.tif`` or ``.vrt``).
    central_bounds
        Bbox in the raster's CRS.

    Returns
    -------
    Path
        Path of the cropped raster. May differ from ``filename`` when the
        input was a ``.vrt`` (extension becomes ``.tif``); callers should
        substitute this back into their file lists.

    """
    from osgeo import gdal

    gdal.UseExceptions()
    src = Path(filename)
    proj_win = (
        central_bounds.left,
        central_bounds.top,
        central_bounds.right,
        central_bounds.bottom,
    )
    if src.suffix == ".tif":
        # In-place rewrite via sibling temp; same path returned.
        tmp = src.with_name(src.stem + ".cropped.tif")
        gdal.Translate(str(tmp), str(src), projWin=proj_win, format="GTiff")
        src.unlink()
        tmp.rename(src)
        return src
    # VRT (or other virtual) input: materialize to a sibling .tif. The
    # original is removed so there's no stale duplicate on disk.
    dst = src.with_suffix(".tif")
    gdal.Translate(str(dst), str(src), projWin=proj_win, format="GTiff")
    src.unlink()
    return dst


def stitch_compressed_slcs(
    comp_slc_dict: dict[str, list[Path]],
    block_bounds: dict[str, BlockBounds],
    output_dir: Filename,
    file_date_fmt: str = "%Y%m%d",
    num_workers: int = 1,
) -> dict[str, list[Path]]:
    """Mosaic per-azimuth-block compressed SLCs into frame-sized files.

    Each azimuth block writes its compressed SLCs at the block's
    ``read_bounds`` extent (central rows + a halo), so the compressed SLCs of
    adjacent blocks overlap in the halo and there is one file *per block* per
    date instead of one frame-sized file. This crops every block's compressed
    SLC down to its (disjoint) ``central_bounds`` and merges same-date files
    across blocks into a single frame-sized compressed SLC -- mirroring how
    the interferograms and other stitching-bound rasters are mosaicked by
    :func:`dolphin.workflows.stitching_bursts.run`.

    Cropping to ``central_bounds`` (rather than keeping the halo) is both
    necessary -- so the disjoint central regions tile the frame without the
    edge-degraded halo pixels winning the overlap -- and correct for the
    next batch's feedback: a frame-sized compressed SLC lets the next run
    re-derive each block's halo from its *neighbor's* central data.

    This is only meaningful when azimuth blocking is active
    (``azimuth_blocks > 1``); the Sentinel-1 / OPERA-burst path never calls it.

    Parameters
    ----------
    comp_slc_dict
        Maps each block id (``"block_00"``, ...) to that block's list of
        compressed SLC paths (one per ministack), as returned by
        ``displacement.run``.
    block_bounds
        Per-block bounds from :func:`split_frame_into_blocks`. A block whose
        ``read_bounds == central_bounds`` (no halo) is merged uncropped.
    output_dir
        Directory to write the frame-sized compressed SLCs into. Should be a
        frame-level directory (outside the ``block_*`` trees so it survives
        per-block pruning).
    file_date_fmt
        ``strftime`` format used to parse the ref/start/end dates from the
        ``compressed_<ref>_<start>_<end>.tif`` filenames.
    num_workers
        Threads used to merge dates in parallel.

    Returns
    -------
    dict[str, list[Path]]
        ``{"compressed_slc": [<frame-sized comp slc per date, sorted>]}``. The
        single synthetic key matches how downstream consumers iterate
        ``comp_slc_dict`` by value (the key itself is not used).

    Notes
    -----
    The per-block compressed SLCs are cropped **in place** (via
    :func:`crop_to_central`), so the full-extent originals are not kept. That
    is safe: by the time this runs, the per-block ministack feedback that
    needed the halo has already completed, and the frame mosaics replace the
    per-block files as the run's output.

    """
    from opera_utils import group_by_date
    from osgeo import gdal
    from tqdm.contrib.concurrent import thread_map

    from dolphin import io, utils

    gdal.UseExceptions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crop each block's compressed SLC down to its (disjoint) central bounds
    # in place -- the per-block files keep distinct paths (one block dir each),
    # so identical ``compressed_<dates>.tif`` basenames still group by date.
    cropped: list[Path] = []
    for burst, comp_slc_files in comp_slc_dict.items():
        bb = block_bounds.get(burst)
        for comp_slc_file in comp_slc_files:
            src = Path(comp_slc_file)
            if bb is None or bb.read_bounds == bb.central_bounds:
                # No halo to strip -- merge the file as-is.
                cropped.append(src)
            else:
                cropped.append(crop_to_central(src, bb.central_bounds))

    # Group the cropped per-block files by their (ref, start, end) date tuple
    # and mosaic each group. The blocks share one frame grid + projection and,
    # after the central crop, tile the frame disjointly -- so this is a pure
    # "drop each tile at its offset" mosaic with no warp, blend, or resample.
    # ``gdal.BuildVRT`` does that virtually (instant, C-level) and a single
    # ``gdal.Translate`` materializes it; that is far cheaper than
    # ``stitching.merge_images`` here, which spawns ``gdal_merge.py`` and adds
    # a bilinear clip pass meant for the general (reprojecting, overlapping)
    # burst case. ``vrtnodata=0`` matches the compressed-SLC nodata.
    grouped = group_by_date(cropped, file_date_fmt=file_date_fmt)

    def _merge_one(item: tuple[tuple, list[Path]]) -> Path:
        dates, files = item
        date_str = utils.format_dates(*dates)
        vrt_path = output_dir / f"compressed_{date_str}.vrt"
        outfile = output_dir / f"compressed_{date_str}.tif"
        gdal.BuildVRT(
            str(vrt_path),
            [str(f) for f in sorted(files)],
            VRTNodata=0,
        )
        gdal.Translate(
            str(outfile),
            str(vrt_path),
            format="GTiff",
            noData=0,
            creationOptions=list(io.DEFAULT_TIFF_OPTIONS),
        )
        vrt_path.unlink(missing_ok=True)
        return outfile

    stitched = thread_map(
        _merge_one,
        list(grouped.items()),
        max_workers=num_workers,
        desc="Stitching compressed SLCs by date",
    )
    return {"compressed_slc": sorted(stitched)}
