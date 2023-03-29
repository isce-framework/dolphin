import itertools
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Union

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

from .config import OPERA_BURST_RE

logger = get_log(__name__)

__all__ = ["group_by_burst", "setup_output_folder"]


def group_by_burst(
    file_list: Sequence[Filename],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
    minimum_slcs: int = 2,
) -> Dict[str, List[Path]]:
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: List[Filename]
        List of paths of CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][dolphin.workflows.config.OPERA_BURST_RE]
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


def setup_output_folder(
    vrt_stack,
    driver: str = "GTiff",
    dtype="complex64",
    start_idx: int = 0,
    strides: Dict[str, int] = {"y": 1, "x": 1},
    creation_options: Optional[List] = None,
    nodata: Optional[float] = 0,
    output_folder: Optional[Path] = None,
) -> List[Path]:
    """Create empty output files for each band after `start_idx` in `vrt_stack`.

    Also creates an empty file for the compressed SLC.
    Used to prepare output for block processing.

    Parameters
    ----------
    vrt_stack : VRTStack
        object containing the current stack of SLCs
    driver : str, optional
        Name of GDAL driver, by default "GTiff"
    dtype : str, optional
        Numpy datatype of output files, by default "complex64"
    start_idx : int, optional
        Index of vrt_stack to begin making output files.
        This should match the ministack index to avoid re-creating the
        past compressed SLCs.
    strides : Dict[str, int], optional
        Strides to use when creating the empty files, by default {"y": 1, "x": 1}
        Larger strides will create smaller output files, computed using
        [dolphin.io.compute_out_shape][]
    creation_options : list, optional
        List of options to pass to the GDAL driver, by default None
    nodata : float, optional
        Nodata value to use for the output files, by default 0.
    output_folder : Path, optional
        Path to output folder, by default None
        If None, will use the same folder as the first SLC in `vrt_stack`

    Returns
    -------
    List[Path]
        List of saved empty files.
    """
    if output_folder is None:
        output_folder = vrt_stack.outfile.parent

    date_strs: List[str] = []
    for d in vrt_stack.dates[start_idx:]:
        if len(d) == 1:
            # normal SLC files will have a single date
            s = d[0].strftime(io.DEFAULT_DATETIME_FORMAT)
        else:
            # Compressed SLCs will have 2 dates in the name marking the start and end
            s = io._format_date_pair(d[0], d[1])
        date_strs.append(s)

    output_files = []
    for filename in date_strs:
        slc_name = Path(filename).stem
        output_path = output_folder / f"{slc_name}.slc.tif"

        io.write_arr(
            arr=None,
            like_filename=vrt_stack.outfile,
            output_name=output_path,
            driver=driver,
            nbands=1,
            dtype=dtype,
            strides=strides,
            nodata=nodata,
            options=creation_options,
        )

        output_files.append(output_path)
    return output_files
