#!/usr/bin/env python
import itertools
import re
from typing import List, Pattern, Union

from dolphin._log import get_log, log_runtime
from dolphin._types import Filename

from . import stitch_and_unwrap, wrapped_phase
from .config import Workflow

# for example, t087_185678_iw2
OPERA_BURST_RE = re.compile(
    r"t(?P<track>\d{3})_(?P<burst_id>\d{6})_(?P<subswath>iw[1-3])"
)


@log_runtime
def run(cfg: Workflow, debug: bool = False):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [Workflow][dolphin.workflows.config.Workflow] object with workflow parameters
    debug : bool, optional
        Enable debug logging, by default False.
    """
    logger = get_log(debug=debug)

    # ###########################
    # 1. Wrapped phase estimation
    # ###########################
    ifg_list = wrapped_phase.run(cfg, debug=debug)

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
    file_list: List[Filename],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
):
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: List[Filename]
        path to folder containing CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [OPERA_BURST_RE][]

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
            [(f, d) for f, d in zip(file_list, burst_ids)],
            # use the date or dates as the key
            key=lambda f_d_tuple: f_d_tuple[1],  # type: ignore
        )
        # Unpack the sorted pairs with new sorted values
        file_list, burst_ids = zip(*file_burst_tups)  # type: ignore
        return file_list

    sorted_file_list = sort_by_burst_id(file_list)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    return grouped_images
