#!/usr/bin/env python
import itertools
import re
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional, Pattern, Sequence, Union

from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.interferogram import VRTInterferogram

from . import stitch_and_unwrap, wrapped_phase
from .config import OPERA_BURST_RE, Workflow


@log_runtime
def run(cfg: Workflow, debug: bool = False, log_file: Optional[str] = None):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [Workflow][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
    log_file : str, optional
        If provided, will log to this file in addition to stderr.
    """
    logger = get_log(debug=debug, filename=log_file)
    logger.debug(pformat(cfg.dict()))

    try:
        grouped_slc_files = _group_by_burst(cfg.inputs.cslc_file_list)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise e
        # Otherwise, we have SLC files which are not OPERA burst files
        grouped_slc_files = {"": cfg.inputs.cslc_file_list}

    if len(grouped_slc_files) > 1:
        logger.info(f"Found SLC files from {len(grouped_slc_files)} bursts")
        wrapped_phase_cfgs = [
            # Include the burst for logging purposes
            (burst, _create_burst_cfg(cfg, burst, grouped_slc_files))
            for burst in grouped_slc_files
        ]
        for _, burst_cfg in wrapped_phase_cfgs:
            burst_cfg.create_dir_tree()
    else:
        wrapped_phase_cfgs = [("", cfg)]
    # ###########################
    # 1. Wrapped phase estimation
    # ###########################
    ifg_list: List[VRTInterferogram] = []
    # Now for each burst, run the wrapped phase estimation
    for burst, burst_cfg in wrapped_phase_cfgs:
        msg = "Running wrapped phase estimation"
        if burst:
            msg += f" for burst {burst}"
        logger.info(msg)
        logger.debug(pformat(burst_cfg.dict()))
        cur_ifg_list = wrapped_phase.run(burst_cfg, debug=debug)
        ifg_list.extend(cur_ifg_list)

    # ###################################
    # 2. Stitch and unwrap interferograms
    # ###################################
    unwrapped_paths = stitch_and_unwrap.run(ifg_list, cfg, debug=debug)

    # ######################################
    # 3. Finalize the output as an HDF5 product
    # ######################################
    # TODO: make the HDF5 product
    logger.info(f"Creating outputs in {cfg.outputs.output_directory}")
    for p in unwrapped_paths:
        # for now, just move the unwrapped results
        #
        # get all the associated header/conncomp files too
        for ext in ["", ".rsc", ".conncomp", ".conncomp.hdr"]:
            name = p.name + ext
            new_name = cfg.outputs.output_directory / name
            logger.info(f"Moving {p} to {new_name}")
            (p.parent / name).rename(new_name)


def _group_by_burst(
    file_list: Sequence[Filename],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
    minimum_slcs: int = 2,
) -> Dict[str, List[Path]]:
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: List[Filename]
        path to folder containing CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [OPERA_BURST_RE][]
    minimum_slcs: int
        Minimum number of SLCs needed to run the workflow for each burst.
        If there are fewer SLCs in a burst, it will be skipped and
        a warning will be logged.

    Returns
    -------
    dict
        key is the burst id of the SLC acquisition
        Value is a list of Paths on that burst:
        {
            't087_185678_iw2': [Path(...), Path(...),],
            't087_185678_iw3': [Path(...),... ],
        }
    """

    def get_burst_id(filename):
        m = re.search(burst_id_fmt, str(filename))
        if not m:
            raise ValueError(f"Could not parse burst id from {filename}")
        return m.group()

    def sort_by_burst_id(file_list):
        """Sort files by burst id."""
        burst_ids = [get_burst_id(f) for f in file_list]
        file_burst_tups = sorted(
            [(Path(f), d) for f, d in zip(file_list, burst_ids)],
            # use the date or dates as the key
            key=lambda f_d_tuple: f_d_tuple[1],  # type: ignore
        )
        # Unpack the sorted pairs with new sorted values
        file_list, burst_ids = zip(*file_burst_tups)  # type: ignore
        return file_list

    logger = get_log()
    sorted_file_list = sort_by_burst_id(file_list)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    # Make sure that each burst has at least the minimum number of SLCs
    out = {}
    for burst_id, slc_list in grouped_images.items():
        if len(slc_list) < minimum_slcs:
            logger.warning(
                f"Skipping burst {burst_id} because it has only {len(slc_list)} SLCs."
                f"Minimum number of SLCs is {minimum_slcs}"
            )
        else:
            out[burst_id] = slc_list
    return out


def _create_burst_cfg(
    cfg: Workflow, burst_id: str, grouped_slc_files: Dict[str, List[Path]]
) -> Workflow:
    excludes = {
        "inputs": {"cslc_file_list"},
        "ps_options": {
            "directory",
            "output_file",
            "amp_dispersion_file",
            "amp_mean_file",
        },
        "phase_linking": {"directory"},
        "interferogram_network": {"directory"},
    }
    cfg_temp_dict = cfg.copy(deep=True, exclude=excludes).dict()

    top_level_scratch = cfg_temp_dict["outputs"]["scratch_directory"]
    new_input_dict = dict(
        inputs={"cslc_file_list": grouped_slc_files[burst_id]},
        outputs={"scratch_directory": top_level_scratch / burst_id},
    )
    # Just update the inputs and the scratch directory
    cfg_temp_dict["inputs"].update(new_input_dict["inputs"])
    cfg_temp_dict["outputs"].update(new_input_dict["outputs"])
    return Workflow(**cfg_temp_dict)
