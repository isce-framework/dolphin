"""Estimate wrapped phase for one ministack of SLCs."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

from dolphin import io, shp, similarity
from dolphin._decorators import atomic_output
from dolphin._types import Filename, HalfWindow, Strides
from dolphin.io import BlockIndices, EagerLoader, StridedBlockManager, VRTStack
from dolphin.masking import load_mask_as_numpy
from dolphin.phase_link import PhaseLinkRuntimeError, compress, run_phase_linking
from dolphin.ps import calc_ps_block
from dolphin.stack import MiniStackInfo
from dolphin.utils import DummyProcessPoolExecutor, grow_nodata_region

from .config import ShpMethod

logger = logging.getLogger("dolphin")

__all__ = ["run_wrapped_phase_single"]


@dataclass
class OutputFile:
    filename: Path
    dtype: DTypeLike
    strides: Optional[dict[str, int]] = None
    nbands: int = 1
    nodata: float = 0


@atomic_output(output_arg="output_folder", is_dir=True)
def run_wrapped_phase_single(
    *,
    vrt_stack: VRTStack,
    ministack: MiniStackInfo,
    output_folder: Filename,
    half_window: dict,
    strides: Optional[dict] = None,
    beta: float = 0.00,
    zero_correlation_threshold: float = 0.0,
    use_evd: bool = False,
    mask_file: Optional[Filename] = None,
    ps_mask_file: Optional[Filename] = None,
    amp_mean_file: Optional[Filename] = None,
    amp_dispersion_file: Optional[Filename] = None,
    shp_method: ShpMethod = ShpMethod.NONE,
    shp_alpha: float = 0.05,
    shp_nslc: Optional[int] = None,
    similarity_nearest_n: int | None = None,
    write_closure_phase: bool = True,
    write_crlb: bool = True,
    block_shape: tuple[int, int] = (512, 512),
    baseline_lag: Optional[int] = None,
    max_workers: int = 1,
    **tqdm_kwargs,
):
    """Estimate wrapped phase for one ministack.

    Output files will all be placed in the provided `output_folder`.
    """
    if strides is None:
        strides = {"x": 1, "y": 1}
    # TODO: extract common stuff between here and sequential
    strides_tup = Strides(y=strides["y"], x=strides["x"])
    half_window_tup = HalfWindow(y=half_window["y"], x=half_window["x"])
    output_folder = Path(output_folder)
    input_slc_files = ministack.file_list
    if len(input_slc_files) != vrt_stack.shape[0]:
        raise ValueError(f"{len(ministack.file_list) = }, but {vrt_stack.shape = }")

    # If we are using a different number of SLCs for the amplitude data,
    # we should note that for the SHP finding algorithms
    if shp_nslc is None:
        shp_nslc = len(input_slc_files)

    logger.info(f"{vrt_stack}: from {ministack.dates[0]} to {ministack.dates[-1]}")

    nrows, ncols = vrt_stack.shape[-2:]

    nodata_mask = _get_nodata_mask(mask_file, nrows, ncols)
    ps_mask = _get_ps_mask(ps_mask_file, nrows, ncols)
    amp_mean, amp_variance = _get_amp_mean_variance(amp_mean_file, amp_dispersion_file)

    xhalf, yhalf = half_window["x"], half_window["y"]

    # If we were passed any compressed SLCs in `input_slc_files`,
    # then we want that index for when we create new compressed SLCs.
    # We skip the old compressed SLCs to create new ones
    first_real_slc_idx = ministack.first_real_slc_idx

    msg = (
        f"Processing {len(input_slc_files) - first_real_slc_idx} SLCs +"
        f" {first_real_slc_idx} compressed SLCs. "
    )
    logger.info(msg)

    # Create the background writer for this ministack
    writer = io.BackgroundBlockWriter()

    logger.info(f"Total stack size (in pixels): {vrt_stack.shape}")
    # Set up the output folder with empty files to write into
    like_filename = vrt_stack.outfile
    phase_linked_slc_files = setup_output_folder(
        ministack=ministack,
        name_generator=_name_slcs,
        strides=strides,
        output_folder=output_folder,
        like_filename=like_filename,
    )

    crlb_output_folder = output_folder / "crlb"
    crlb_output_folder.mkdir(exist_ok=True)
    phase_linked_crlb_files: list[Path] = []
    closure_phase_files: list[Path] = []
    if write_crlb:
        phase_linked_crlb_files = setup_output_folder(
            ministack=ministack,
            name_generator=_name_crlbs,
            strides=strides,
            dtype="float32",
            output_folder=crlb_output_folder,
            like_filename=like_filename,
        )

    if write_closure_phase:
        closure_phases_output_folder = output_folder / "closure_phases"
        closure_phases_output_folder.mkdir(exist_ok=True)
        closure_phase_files = setup_output_folder(
            ministack=ministack,
            name_generator=_name_closure_phases,
            strides=strides,
            dtype="float32",
            output_folder=closure_phases_output_folder,
            like_filename=like_filename,
        )

    comp_slc_info = ministack.get_compressed_slc_info()

    # Use the real-SLC date range for output file naming
    start_end = ministack.real_slc_date_range_str
    output_files: dict[str, OutputFile] = {
        # The compressed SLC does not used strides, but has extra band for dispersion
        "compressed_slc": OutputFile(
            output_folder / comp_slc_info.filename, np.complex64, nbands=2
        ),
        # but all the rest do:
        "temporal_coherence": OutputFile(
            output_folder / f"temporal_coherence_{start_end}.tif", np.float32, strides
        ),
        "shp_counts": OutputFile(
            output_folder / f"shp_counts_{start_end}.tif", np.uint16, strides
        ),
        "eigenvalues": OutputFile(
            output_folder / f"eigenvalues_{start_end}.tif", np.float32, strides
        ),
        "estimator": OutputFile(
            output_folder / f"estimator_{start_end}.tif",
            np.int8,
            strides,
            nodata=255,
        ),
    }

    for op in output_files.values():
        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=op.filename,
            dtype=op.dtype,
            strides=op.strides,
            nbands=op.nbands,
            nodata=op.nodata,
        )

    # Iterate over the output grid
    block_manager = StridedBlockManager(
        arr_shape=(nrows, ncols),
        block_shape=block_shape,
        strides=strides_tup,
        half_window=half_window_tup,
    )
    # Set up the background loader
    loader = EagerLoader(reader=vrt_stack, block_shape=block_shape)
    # Queue all input slices, skip ones that are all nodata
    blocks: list[
        tuple[BlockIndices, BlockIndices, BlockIndices, BlockIndices, BlockIndices]
    ] = []
    # Queue all input slices, skip ones that are all nodata
    for b in block_manager.iter_blocks():
        in_rows, in_cols = b[2]
        # nodata_mask is numpy convention: True for bad (masked).
        if nodata_mask[in_rows, in_cols].all():
            continue
        # loader.queue_read(in_rows, in_cols)
        blocks.append(b)
    ###########################
    write_lock = Lock()
    read_lock = Lock()

    Executor = ThreadPoolExecutor if max_workers > 1 else DummyProcessPoolExecutor
    pbar = tqdm(total=len(blocks), **tqdm_kwargs)

    def _process_block(
        block: tuple[
            BlockIndices, BlockIndices, BlockIndices, BlockIndices, BlockIndices
        ],
    ):
        (
            (out_rows, out_cols),
            (out_trim_rows, out_trim_cols),
            (in_rows, in_cols),
            (in_no_pad_rows, in_no_pad_cols),
            (in_trim_rows, in_trim_cols),
        ) = block
        with read_lock:
            cur_data, _ = loader.read(in_rows, in_cols)
        if np.all(cur_data == 0) or np.isnan(cur_data).all():
            return block, None, None, None

        cur_data = cur_data.astype(np.complex64)

        # Only actually compute if we need this one
        amp_stack = np.abs(cur_data) if shp_method == "ks" else None

        # Compute the neighbor_arrays for this block
        neighbor_arrays = shp.estimate_neighbors(
            halfwin_rowcol=(yhalf, xhalf),
            alpha=shp_alpha,
            strides=Strides(y=strides_tup[0], x=strides_tup[1]),
            mean=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
            var=amp_variance[in_rows, in_cols] if amp_variance is not None else None,
            nslc=shp_nslc,
            amp_stack=amp_stack,
            method=shp_method,
        )
        try:
            pl_output = run_phase_linking(
                cur_data,
                half_window=half_window_tup,
                strides=strides_tup,
                use_evd=use_evd,
                beta=beta,
                zero_correlation_threshold=zero_correlation_threshold,
                reference_idx=ministack.output_reference_idx,
                nodata_mask=nodata_mask[in_rows, in_cols],
                ps_mask=ps_mask[in_rows, in_cols],
                neighbor_arrays=neighbor_arrays,
                baseline_lag=baseline_lag,
                avg_mag=amp_mean[in_rows, in_cols] if amp_mean is not None else None,
                first_real_slc_idx=ministack.first_real_slc_idx,
                compute_crlb=write_crlb,
            )
        except PhaseLinkRuntimeError as e:
            # note: this is a warning instead of info, since it should
            # get caught at the "skip_empty" step
            msg = f"At block {in_rows.start}, {in_cols.start}: {e}"
            if "are all NaNs" in e.args[0]:
                # Some SLCs in the ministack are all NaNs
                # This happens from a shifting burst window near the edges,
                # and seems to cause no issues
                logger.debug(msg)
            else:
                logger.warning(msg)
            return block, None, None, None

        # Fill in the nan values with 0
        np.nan_to_num(pl_output.cpx_phase, copy=False)
        np.nan_to_num(pl_output.crlb_std_dev, copy=False)
        np.nan_to_num(pl_output.temp_coh, copy=False)

        # Compress the ministack using only the non-compressed SLCs
        # Get the mean to set as pixel magnitudes
        abs_stack = np.abs(cur_data[first_real_slc_idx:, in_trim_rows, in_trim_cols])
        cur_data_mean, cur_amp_dispersion, _ = calc_ps_block(abs_stack)
        cur_comp_slc = compress(
            # Get the inner portion of the full-res SLC data
            cur_data[:, in_trim_rows, in_trim_cols],
            pl_output.cpx_phase[:, out_trim_rows, out_trim_cols],
            first_real_slc_idx=first_real_slc_idx,
            slc_mean=cur_data_mean,
            reference_idx=ministack.compressed_reference_idx,
        )

        # Save each of the MLE estimates (ignoring those corresponding to
        # compressed SLCs indexes)
        assert len(pl_output.cpx_phase[first_real_slc_idx:]) == len(
            phase_linked_slc_files
        )
        # ### Save results ###
        with write_lock:
            # ### Save results ###
            for img, f in zip(
                pl_output.cpx_phase[first_real_slc_idx:, out_trim_rows, out_trim_cols],
                phase_linked_slc_files,
                strict=True,
            ):
                writer.queue_write(img, f, out_rows.start, out_cols.start)

            if write_crlb:
                for img, f in zip(
                    pl_output.crlb_std_dev[
                        first_real_slc_idx:, out_trim_rows, out_trim_cols
                    ],
                    phase_linked_crlb_files,
                    strict=True,
                ):
                    writer.queue_write(img, f, out_rows.start, out_cols.start)

            if write_closure_phase:
                # Save closure phases (N-2 images for N dates)
                for i, closure_file in enumerate(closure_phase_files):
                    closure_img = pl_output.closure_phases[
                        out_trim_rows, out_trim_cols, i
                    ]
                    writer.queue_write(
                        closure_img, closure_file, out_rows.start, out_cols.start
                    )

            # Save the compressed SLC block
            writer.queue_write(
                cur_comp_slc,
                output_files["compressed_slc"].filename,
                in_no_pad_rows.start,
                in_no_pad_cols.start,
                band=1,
            )
            # Save the amplitude dispersion of the real SLC data
            writer.queue_write(
                cur_amp_dispersion,
                output_files["compressed_slc"].filename,
                in_no_pad_rows.start,
                in_no_pad_cols.start,
                band=2,
            )

            # All other outputs are strided (smaller in size)
            out_datas: dict[str, np.ndarray] = {
                "temporal_coherence": pl_output.temp_coh,
                "shp_counts": pl_output.shp_counts,
                "eigenvalues": pl_output.eigenvalues,
                "estimator": pl_output.estimator,
            }
            for key, data in out_datas.items():
                output_file = output_files[key]
                trimmed_data = data[out_trim_rows, out_trim_cols]

                writer.queue_write(
                    # Erode the edge pixels before writing:
                    grow_nodata_region(
                        trimmed_data, nodata=output_file.nodata, n_pixels=2, copy=True
                    ),
                    output_file.filename,
                    out_rows.start,
                    out_cols.start,
                )
            pbar.update()

    with Executor(max_workers) as exc:
        # Consume all blocks from the `.map` call
        deque(exc.map(_process_block, blocks))

    # Block until all the writers for this ministack have finished
    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    logger.info(f"Finished ministack of size {vrt_stack.shape}.")
    loader.notify_finished()

    logger.info("Repacking phase linking outputs for more compression")
    io.repack_rasters(phase_linked_slc_files, keep_bits=12)

    logger.info("Creating similarity raster on outputs")
    similarity.create_similarities(
        phase_linked_slc_files,
        output_file=output_folder / f"similarity_{start_end}.tif",
        num_threads=1,
        add_overviews=False,
        nearest_n=similarity_nearest_n,
        block_shape=block_shape,
    )

    if write_crlb:
        logger.info("Repacking CRLB files for more compression")
        # CRLB needs only low precision output
        io.repack_rasters(phase_linked_crlb_files, use_16_bits=True)
    if write_closure_phase:
        logger.info("Repacking closure phase files for more compression")
        io.repack_rasters(closure_phase_files, keep_bits=10)

    written_comp_slc = output_files["compressed_slc"]
    ccslc_info = ministack.get_compressed_slc_info()
    ccslc_info.write_metadata(output_file=written_comp_slc.filename)
    # TODO: Does it make sense to return anything from this?
    # or just allow user to search through the `output_folder` they provided?


def _get_nodata_mask(
    mask_file: Optional[Filename],
    nrows: int,
    ncols: int,
) -> np.ndarray:
    if mask_file is not None:
        return load_mask_as_numpy(mask_file)
    else:
        return np.zeros((nrows, ncols), dtype=bool)


def _get_ps_mask(
    ps_mask_file: Optional[Filename], nrows: int, ncols: int
) -> np.ndarray:
    if ps_mask_file is not None:
        ps_mask = io.load_gdal(ps_mask_file, masked=True)
        # Fill the nodata values with false
        ps_mask = ps_mask.astype(bool).filled(False)
    else:
        ps_mask = np.zeros((nrows, ncols), dtype=bool)
    return ps_mask


def _get_amp_mean_variance(
    amp_mean_file: Optional[Filename],
    amp_dispersion_file: Optional[Filename],
) -> tuple[np.ndarray, np.ndarray]:
    if amp_mean_file is not None and amp_dispersion_file is not None:
        # Note: have to fill, since numba (as of 0.57) can't do masked arrays
        amp_mean = io.load_gdal(amp_mean_file, masked=True).filled(np.nan)
        amp_dispersion = io.load_gdal(amp_dispersion_file, masked=True).filled(np.nan)
        # convert back to variance from dispersion: amp_disp = std_dev / mean
        amp_variance = (amp_dispersion * amp_mean) ** 2
    else:
        amp_mean = amp_variance = None

    return amp_mean, amp_variance


def _name_slcs(ministack: MiniStackInfo) -> list[str]:
    """Generate SLC filenames for the ministack."""
    start_idx = ministack.first_real_slc_idx
    date_strs = ministack.get_date_str_list()[start_idx:]
    return [f"{Path(d).stem}.slc.tif" for d in date_strs]


def _name_crlbs(ministack: MiniStackInfo) -> list[str]:
    """Generate CRLB filenames for the ministack."""
    start_idx = ministack.first_real_slc_idx
    date_strs = ministack.get_date_str_list()[start_idx:]
    return [f"crlb_{Path(d).stem}.tif" for d in date_strs]


def _name_closure_phases(ministack: MiniStackInfo) -> list[str]:
    """Generate closure phase triplet filenames for the ministack."""
    date_strs = ministack.get_date_str_list()
    # Get only the first date in case of compressed
    date_strs = [d.split("_")[0] for d in date_strs]
    # Create triplets
    num_closure_phases = len(date_strs) - 2
    date_triplets = ["_".join(date_strs[i : i + 3]) for i in range(num_closure_phases)]
    return [f"closure_phase_{triplet}.tif" for triplet in date_triplets]


def setup_output_folder(
    ministack: MiniStackInfo,
    name_generator: Callable[[MiniStackInfo], list[str]],
    driver: str = "GTiff",
    dtype="complex64",
    like_filename: Optional[Filename] = None,
    strides: Optional[dict[str, int]] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> list[Path]:
    """Create empty raster files in the output folder.

    Used to prepare outputs for phase linking, CRLB estimates,
    and closure phase triplets.

    Parameters
    ----------
    ministack : MiniStackInfo
        [dolphin.stack.MiniStackInfo][] object for the current batch of SLCs
    name_generator : Callable[[MiniStackInfo], list[str]]
        Function that generates the names of the output files for a given
        ministack.
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    like_filename : Filename, optional
        Filename to use for getting the shape/GDAL metadata of the output files.
        If None, will use the first SLC in `vrt_stack`
    strides : dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    list[Path]
        list of saved empty files for the outputs of phase linking

    """
    """Create empty output files using custom filename generation logic."""
    if strides is None:
        strides = {"y": 1, "x": 1}
    if output_folder is None:
        output_folder = ministack.output_folder
    # Note: during the workflow, the ministack.output_folder is different than
    # the `run_wrapped_phase_single` argument `output_folder`.
    # The latter is the tempdir made by @atomic_output
    output_folder.mkdir(exist_ok=True, parents=True)

    filenames = name_generator(ministack)
    output_files = []
    for filename in filenames:
        output_path = output_folder / filename

        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
        )
        output_files.append(output_path)

    return output_files
