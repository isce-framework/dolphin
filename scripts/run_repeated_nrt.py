#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any, Mapping, Sequence

from dolphin import io, ps, stack, utils
from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.workflows import s1_disp
from dolphin.workflows._utils import group_by_burst, make_nodata_mask
from dolphin.workflows.config import (
    OPERA_DATASET_NAME,
    InterferogramNetworkType,
    ShpMethod,
    Workflow,
    WorkflowName,
)

logger = get_log("dolphin.run_repeated_nrt")


def _create_cfg(
    *,
    slc_files: Sequence[Filename],
    half_window_size: tuple[int, int] = (11, 5),
    first_ministack: bool = False,
    run_unwrap: bool = False,
    shp_method: ShpMethod = ShpMethod.GLRT,
    amplitude_mean_files: Sequence[Filename] = [],
    amplitude_dispersion_files: Sequence[Filename] = [],
    strides: Mapping[str, int] = {"x": 6, "y": 3},
    work_dir: Path = Path("."),
    n_parallel_bursts: int = 1,
):
    # strides = {"x": 1, "y": 1}
    interferogram_network: dict[str, Any]
    if first_ministack:
        interferogram_network = dict(
            network_type=InterferogramNetworkType.SINGLE_REFERENCE
        )
        workflow_name = WorkflowName.STACK
    else:
        interferogram_network = dict(
            network_type=InterferogramNetworkType.MANUAL_INDEX,
            indexes=[(0, -1)],
        )
        workflow_name = WorkflowName.SINGLE

    cfg = Workflow(
        # Things that change with each workflow run
        cslc_file_list=slc_files,
        interferogram_network=interferogram_network,
        amplitude_mean_files=amplitude_mean_files,
        amplitude_dispersion_files=amplitude_dispersion_files,
        # Configurable from CLI inputs:
        output_options=dict(
            strides=strides,
        ),
        phase_linking=dict(
            ministack_size=1000,  # for single update, process in one ministack
            half_window={"x": half_window_size[0], "y": half_window_size[1]},
            shp_method=shp_method,
        ),
        scratch_directory=work_dir / "scratch",
        output_directory=work_dir / "output",
        worker_settings=dict(
            #     block_size_gb=block_size_gb,
            n_parallel_bursts=n_parallel_bursts,
            n_workers=4,
            threads_per_worker=8,
        ),
        #     ps_options=dict(
        #         amp_dispersion_threshold=amp_dispersion_threshold,
        #     ),
        #     log_file=log_file,
        # )
        # Definite hard coded things
        unwrap_options=dict(
            unwrap_method="snaphu",
            run_unwrap=run_unwrap,
            # CHANGEME: or else run in background somehow?
        ),
        save_compressed_slc=True,  # always save, and only sometimes will we grab it
        workflow_name=workflow_name,
    )
    return cfg


def get_cli_args() -> argparse.Namespace:
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Repeatedly run the dolphin single update mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # https://docs.python.org/3/library/argparse.html#fromfile-prefix-chars
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--slc-files",
        nargs=argparse.ONE_OR_MORE,
        help="List the paths of all SLC files to include.",
        required=True,
    )
    parser.add_argument(
        "-ms",
        "--ministack-size",
        type=int,
        default=10,
        help="Strides/decimation factor (x, y) (in pixels) to use when determining",
    )
    parser.add_argument(
        "-hw",
        "--half-window-size",
        type=int,
        nargs=2,
        default=(11, 5),
        metavar=("X", "Y"),
        help="Half window size for the phase linking algorithm",
    )
    parser.add_argument(
        "--shp-method",
        type=ShpMethod,
        choices=[s.value for s in ShpMethod],
        default=ShpMethod.GLRT,
        help="Method used to calculate the SHP.",
    )
    parser.add_argument(
        "--run-unwrap",
        action="store_true",
        help="Run the unwrapping stack after phase linking.",
    )
    parser.add_argument(
        "-j",
        "--n-parallel-bursts",
        type=int,
        default=1,
        help="Number of parallel bursts to process.",
    )
    parser.add_argument(
        "--pre-compute",
        action="store_true",
        help=(
            "Run the amplitude mean/dispersion pre-compute step (not the main"
            " workflow)."
        ),
    )
    return parser.parse_args()


@log_runtime
def compute_ps_files(
    burst_grouped_slc_files: Mapping[str, Sequence[Filename]],
    burst_to_nodata_mask: Mapping[str, Filename],
    # max_workers: int = 3,
    ps_stack_size: int = 60,
    output_folder: Path = Path("precomputed_ps"),
):
    """Compute the mean/DA/PS files for each burst group."""
    all_amp_files, all_disp_files, all_ps_files = [], [], []
    # future_burst_dict = {}
    # with ThreadPoolExecutor(max_workers=max_workers) as exc:
    for burst, file_list in burst_grouped_slc_files.items():
        nodata_mask_file = burst_to_nodata_mask[burst]
        # fut = exc.submit(
        # _compute_burst_ps_files,
        amp_files, disp_files, ps_files = _compute_burst_ps_files(
            burst,
            file_list,
            nodata_mask_file=nodata_mask_file,
            ps_stack_size=ps_stack_size,
            output_folder=output_folder,
            # show_progress=False,
        )
        # future_burst_dict[fut] = burst

        # for future in as_completed(future_burst_dict.keys()):
        # burst = future_burst_dict[future]
        # amp_files, disp_files, ps_files = future.result()

        all_amp_files.extend(amp_files)
        all_disp_files.extend(disp_files)
        all_ps_files.extend(ps_files)

        logger.info(f"Done with {burst}")

    return all_amp_files, all_disp_files, all_ps_files


@log_runtime
def _compute_burst_ps_files(
    burst: str,
    file_list_all: Sequence[Filename],
    nodata_mask_file: Filename,
    ps_stack_size: int = 60,
    output_folder: Path = Path("precomputed_ps"),
) -> tuple[list[Path], list[Path], list[Path]]:
    """Pre-compute the PS files (mean / amp. dispersion) for one burst."""
    logger.info(f"Computing PS files for {burst} into {output_folder}")
    vrt_all = stack.VRTStack(
        file_list_all, subdataset=OPERA_DATASET_NAME, write_file=False
    )
    # logger.info("Created total vrt")
    date_list_all = vrt_all.dates

    nodata_mask = io.load_gdal(nodata_mask_file, masked=True).astype(bool).filled(False)
    # invert the mask so 1s are the missing data pixels
    nodata_mask = ~nodata_mask

    # TODO: fixed number of PS files? fixed time window?
    amp_files, disp_files, ps_files = [], [], []
    for full_stack_idx in range(0, len(file_list_all), ps_stack_size):
        cur_slice = slice(full_stack_idx, full_stack_idx + ps_stack_size)
        cur_files = file_list_all[cur_slice]
        cur_dates = date_list_all[cur_slice]

        # Make the current ministack output folder using the start/end dates
        d0 = cur_dates[0][0]
        d1 = cur_dates[-1][0]
        start_end = io._format_date_pair(d0, d1)
        basename = f"{burst}_{start_end}"

        # output_folder = output_folder / start_end
        output_folder.mkdir(parents=True, exist_ok=True)
        cur_vrt = stack.VRTStack(
            cur_files,
            outfile=output_folder / f"{basename}.vrt",
            subdataset=OPERA_DATASET_NAME,
        )
        cur_ps_file = (output_folder / f"{basename}_ps_pixels.tif").resolve()
        cur_amp_mean = (output_folder / f"{basename}_amp_mean.tif").resolve()
        cur_amp_dispersion = (
            output_folder / f"{basename}_amp_dispersion.tif"
        ).resolve()
        if not all(f.exists() for f in [cur_ps_file, cur_amp_mean, cur_amp_dispersion]):
            ps.create_ps(
                slc_vrt_file=cur_vrt,
                output_amp_mean_file=cur_amp_mean,
                output_amp_dispersion_file=cur_amp_dispersion,
                output_file=cur_ps_file,
                nodata_mask=nodata_mask,
                block_size_gb=0.2,
            )
        else:
            logger.info(f"Skipping existing {basename} files in {output_folder}")
        amp_files.append(cur_amp_mean)
        disp_files.append(cur_amp_dispersion)
        ps_files.append(cur_ps_file)
        logger.info(f"Finished with PS processing for {burst}")
    return amp_files, disp_files, ps_files


def create_nodata_masks(
    # date_grouped_slc_files: dict[tuple[datetime.date], list[Filename]],
    burst_grouped_slc_files: Mapping[str, Sequence[Filename]],
    buffer_pixels: int = 30,
    max_workers: int = 3,
    output_folder: Path = Path("nodata_masks"),
):
    """Create the nodata binary masks for each burst."""
    output_folder.mkdir(exist_ok=True, parents=True)
    futures = []
    out_burst_to_file = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        for burst, file_list in burst_grouped_slc_files.items():
            outfile = output_folder / f"{burst}.tif"
            fut = exc.submit(
                make_nodata_mask,
                file_list,
                outfile,
                buffer_pixels=buffer_pixels,
            )
            futures.append(fut)
            out_burst_to_file[burst] = outfile
        for fut in futures:
            fut.result()
    return out_burst_to_file


def _form_burst_vrt_stacks(
    burst_grouped_slc_files: Mapping[str, Sequence[Filename]]
) -> dict[str, stack.VRTStack]:
    logger.info("For each burst, creating a VRTStack...")
    # Each burst needs to be the same size
    burst_to_vrt_stack = {}
    for b, file_list in burst_grouped_slc_files.items():
        logger.info(f"Checking {len(file_list)} files for {b}")
        outfile = Path(f"slc_stack_{b}.vrt")
        if not outfile.exists():
            vrt = stack.VRTStack(
                file_list, subdataset=OPERA_DATASET_NAME, outfile=outfile
            )
        else:
            vrt = stack.VRTStack.from_vrt_file(outfile, skip_size_check=True)
        burst_to_vrt_stack[b] = vrt
    logger.info("Done.")
    return burst_to_vrt_stack


@log_runtime
def precompute_ps_files(arg_dict) -> None:
    """Run the pre-compute step to get means/amp. dispersion for each burst."""
    all_slc_files = arg_dict.pop("slc_files")

    burst_grouped_slc_files = group_by_burst(all_slc_files)
    #  {'t173_370312_iw2': [PosixPath('t173_370312_iw2_20170203.h5'),... ] }
    date_grouped_slc_files = utils.group_by_date(all_slc_files)
    #  { (datetime.date(2017, 5, 22),) : [PosixPath('t173_370311_iw1_20170522.h5'), ] }
    logger.info(f"Found {len(all_slc_files)} total SLC files")
    logger.info(f"  {len(date_grouped_slc_files)} unique dates,")
    logger.info(f"  {len(burst_grouped_slc_files)} unique bursts.")

    burst_to_nodata_mask = create_nodata_masks(burst_grouped_slc_files)

    all_amp_files, all_disp_files, all_ps_files = compute_ps_files(
        burst_grouped_slc_files, burst_to_nodata_mask
    )


def _get_all_slc_files(
    burst_to_file_list: Mapping[str, Sequence[Filename]], start_idx: int, end_idx: int
) -> list[Filename]:
    return list(
        chain.from_iterable(
            [file_list[start_idx:end_idx] for file_list in burst_to_file_list.values()]
        )
    )


def _run_one_stack(
    slc_idx_start: int,
    slc_idx_end: int,
    ministack_size: int,
    burst_to_file_list: dict[str, Sequence[Filename]],
    comp_slc_files: list[Filename],
    all_amp_files: Sequence[Filename],
    all_disp_files: Sequence[Filename],
):
    cur_path = Path(f"stack_{slc_idx_start}_{slc_idx_end}")
    cur_path.mkdir(exist_ok=True)

    logger.info(f"***** START: {cur_path} *****")
    # Get the nearest amplitude mean/dispersion files
    cur_slc_files = _get_all_slc_files(burst_to_file_list, slc_idx_start, slc_idx_end)
    cfg = _create_cfg(
        slc_files=comp_slc_files + cur_slc_files,
        amplitude_mean_files=all_amp_files,
        amplitude_dispersion_files=all_disp_files,
        work_dir=cur_path,
        **arg_dict,
    )
    cfg.to_yaml(cur_path / "dolphin_config.yaml")
    s1_disp.run(cfg)

    # On the step before we hit double `ministack_size`,
    # archive, shrink, and pull another compressed SLC to replace.
    stack_size = slc_idx_end - slc_idx_start
    max_stack_size = 2 * ministack_size - 1  # Size at which we archive/shrink
    if stack_size == max_stack_size:
        # time to shrink!
        # Get the compressed SLC that was output
        comp_slc_path = (cur_path / "output/compressed_slcs/").resolve()
        new_comp_slcs = list(comp_slc_path.glob("*.h5"))
    else:
        new_comp_slcs = []

    logger.info(f"***** END: {cur_path} *****")
    return new_comp_slcs


@log_runtime
def main(arg_dict: dict) -> None:
    """Get the command line arguments and run the workflow."""
    arg_dict = vars(args)
    ministack_size = arg_dict.pop("ministack_size")
    # TODO: verify this is fine to sort them by date?
    all_slc_files = sorted(arg_dict.pop("slc_files"))
    logger.info(f"Found {len(all_slc_files)} total SLC files")

    # format of `group_by_burst`:
    #   {'t173_370312_iw2': [PosixPath('t173_370312_iw2_20170203.h5'),... ] }
    burst_grouped_slc_files = group_by_burst(all_slc_files)
    num_bursts = len(burst_grouped_slc_files)
    logger.info(f"  {num_bursts} unique bursts.")
    # format of `group_by_date`:
    #  { (datetime.date(2017, 5, 22),) : [PosixPath('t173_370311_iw1_20170522.h5'), ] }
    date_grouped_slc_files = utils.group_by_date(all_slc_files)
    num_dates = len(date_grouped_slc_files)
    logger.info(f"  {num_dates} unique dates,")

    burst_to_vrt_stack = _form_burst_vrt_stacks(
        burst_grouped_slc_files=burst_grouped_slc_files
    )
    burst_to_file_list = {b: v.file_list for b, v in burst_to_vrt_stack.items()}

    # Get the pre-compted PS files (assuming --pre-compute has been run)
    all_amp_files = sorted(Path("precomputed_ps/").resolve().glob("*_amp_mean.tif"))
    all_disp_files = sorted(
        Path("precomputed_ps/").resolve().glob("*_amp_dispersion.tif")
    )

    slc_idx_start = 0
    slc_idx_end = ministack_size
    # max_stack_size = 2 * ministack_size - 1  # Size at which we archive/shrink
    cur_path = Path(f"stack_{slc_idx_start}_{slc_idx_end}")
    cur_path.mkdir(exist_ok=True)

    # TODO: how to make it shift when the year changes for PS files

    # Make the first ministack
    cur_slc_files = _get_all_slc_files(burst_to_file_list, slc_idx_start, slc_idx_end)
    cfg = _create_cfg(
        slc_files=cur_slc_files,
        first_ministack=True,
        amplitude_mean_files=all_amp_files,
        amplitude_dispersion_files=all_disp_files,
        work_dir=cur_path,
        **arg_dict,
    )
    cfg.to_yaml(cur_path / "dolphin_config.yaml")
    s1_disp.run(cfg)

    # Rest of mini stacks in incremental-mode
    comp_slc_files: list[Path] = []
    slc_idx_end = ministack_size + 1

    slc_idx_start, slc_idx_end = 10, 28
    while slc_idx_end <= num_dates:
        # we have to wait for the shrink-and-archive jobs before continuing
        new_comp_slcs = _run_one_stack(
            slc_idx_start,
            slc_idx_end,
            ministack_size,
            burst_to_file_list,
            comp_slc_files,
            all_amp_files,
            all_disp_files,
        )
        logger.info(
            f"{len(new_comp_slcs)} comp slcs from stack_{slc_idx_start}_{slc_idx_end}"
        )
        comp_slc_files.extend(new_comp_slcs)
        slc_idx_end += 1
        if len(new_comp_slcs) > 0:
            # Move the front idx up by one ministack
            slc_idx_start += ministack_size


if __name__ == "__main__":
    args = get_cli_args()
    arg_dict = vars(args)
    if arg_dict.pop("pre_compute"):
        precompute_ps_files(arg_dict)
    else:
        main(arg_dict)
