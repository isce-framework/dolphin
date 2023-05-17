#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

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
    shp_method: ShpMethod = ShpMethod.GLRT,
):
    strides = {"x": 6, "y": 3}
    if first_ministack:
        interferogram_network = dict(
            network_type=InterferogramNetworkType.SINGLE_REFERENCE
        )
    else:
        interferogram_network = dict(
            network_type=InterferogramNetworkType.MANUAL_INDEX,
            indexes=[(0, -1)],
        )

    cfg = Workflow(
        # Things that change with each workflow run
        cslc_file_list=slc_files,
        interferogram_network=interferogram_network,
        amplitude_mean_files=[],
        amplitude_dispersion_files=[],
        # Configurable from CLI inputs:
        output_options=dict(
            strides=strides,
        ),
        unwrap_options=dict(
            unwrap_method="snaphu",
        ),
        phase_linking=dict(
            ministack_size=500,  # for single update, process in one ministack
            half_window={"x": half_window_size[0], "y": half_window_size[1]},
            shp_method=shp_method,
        ),
        # worker_settings=dict(
        #     block_size_gb=block_size_gb,
        #     n_workers=n_workers,
        #     threads_per_worker=threads_per_worker,
        # ),
        #     ps_options=dict(
        #         amp_dispersion_threshold=amp_dispersion_threshold,
        #     ),
        #     log_file=log_file,
        # )
        # Definite hard coded things
        workflow_name=WorkflowName.SINGLE,
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
        default=15,
        help="Strides/decimation factor (x, y) (in pixels) to use when determining",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Dont open all GSLCs to verify they are valid.",
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
    return parser.parse_args()


@log_runtime
def compute_ps_files(
    burst_grouped_slc_files: dict[str, list[Filename]],
    burst_to_nodata_mask: dict[str, Filename],
    max_workers: int = 3,
    ps_stack_size: int = 60,
    output_folder: Path = Path("precomputed_ps"),
):
    """Compute the mean/DA/PS files for each burst group."""
    all_amp_files, all_disp_files, all_ps_files = [], [], []
    future_burst_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        for burst, file_list in burst_grouped_slc_files.items():
            nodata_mask_file = burst_to_nodata_mask[burst]
            fut = exc.submit(
                _compute_burst_ps_files,
                # amp_files, disp_files, ps_files = compute_burst_ps_files(
                burst,
                file_list,
                nodata_mask_file=nodata_mask_file,
                ps_stack_size=ps_stack_size,
                output_folder=output_folder,
            )
            future_burst_dict[fut] = burst

        for future in as_completed(future_burst_dict.keys()):
            burst = future_burst_dict[future]
            amp_files, disp_files, ps_files = future.result()

            all_amp_files.extend(amp_files)
            all_disp_files.extend(disp_files)
            all_ps_files.extend(ps_files)

        logger.info(f"Done with {burst}")

    return all_amp_files, all_disp_files, all_ps_files


@log_runtime
def _compute_burst_ps_files(
    burst: str,
    file_list_all: list[Filename],
    nodata_mask_file: Optional[Filename] = None,
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
        cur_ps_file = output_folder / f"{basename}_ps_pixels.tif"
        cur_amp_mean = output_folder / f"{basename}_amp_mean.tif"
        cur_amp_dispersion = output_folder / f"{basename}_amp_dispersion.tif"
        if not all(f.exists() for f in [cur_ps_file, cur_amp_mean, cur_amp_dispersion]):
            ps.create_ps(
                slc_vrt_file=cur_vrt,
                output_amp_mean_file=cur_amp_mean,
                output_amp_dispersion_file=cur_amp_dispersion,
                output_file=cur_ps_file,
                nodata_mask=nodata_mask,
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
    burst_grouped_slc_files: dict[str, list[Filename]],
    buffer_pixels: int = 30,
    max_workers: int = 3,
    output_folder: Path = Path("nodata_masks"),
):
    """Create the nodata binary masks for each burst."""
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


def main() -> None:
    """Get the command line arguments and run the workflow."""
    args = get_cli_args()
    arg_dict = vars(args)
    ministack_size = arg_dict.pop("ministack_size")
    all_slc_files = arg_dict.pop("slc_files")

    burst_grouped_slc_files = group_by_burst(all_slc_files)
    #  {'t173_370312_iw2': [PosixPath('t173_370312_iw2_20170203.h5'),... ] }
    date_grouped_slc_files = utils.group_by_date(all_slc_files)
    #  { (datetime.date(2017, 5, 22),) : [PosixPath('t173_370311_iw1_20170522.h5'), ] }
    logger.info(f"Found {len(all_slc_files)} total SLC files")
    logger.info(f"  {len(date_grouped_slc_files)} unique dates,")
    logger.info(f"  {len(burst_grouped_slc_files)} unique bursts.")

    burst_to_nodata_mask = create_nodata_masks(burst_grouped_slc_files)

    if not args.skip_verify:
        logger.info(
            f"Verifying all {len(all_slc_files)} total SLC files by creating a VRT"
            " stack..."
        )
        stack.VRTStack(all_slc_files, subdataset=OPERA_DATASET_NAME, write_file=False)
        logger.info("Done.")

    compute_ps_files(burst_grouped_slc_files, burst_to_nodata_mask)

    # Make the first ministack
    cur_slc_files = all_slc_files[:ministack_size]
    cfg = _create_cfg(
        slc_files=cur_slc_files,
        first_ministack=True,
        **arg_dict,
    )

    # logger.info(f"Saving configuration to {str(outfile)}")
    # cfg.to_yaml(outfile)
    s1_disp.run(cfg)


if __name__ == "__main__":
    main()
