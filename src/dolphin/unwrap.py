import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal

from dolphin._log import get_log, log_runtime
from dolphin._types import Filename
from dolphin.io import copy_projection, get_raster_xysize
from dolphin.utils import progress

logger = get_log(__name__)

gdal.UseExceptions()


def unwrap(
    ifg_file: Filename,
    out_file: Filename,
    cor_file: Optional[Filename] = None,
    mask_file: Optional[Filename] = None,
    do_tile: bool = False,
    init_method: str = "mcf",
    looks: Tuple[int, int] = (5, 1),
    alt_line_data: bool = True,
) -> subprocess.CompletedProcess:
    """Unwrap a single interferogram using snaphu.

    Parameters
    ----------
    ifg_file : Filename
        The interferogram file to unwrap.
    out_file : Filename
        The output file to save the unwrapped interferogram.
    cor_file : Optional[Filename], optional
        The coherence file to use for unwrapping, by default None
    mask_file : Optional[Filename], optional
        The mask file to use for unwrapping, by default None.
    do_tile : bool, optional
        Whether to tile the unwrapping, by default False.
    init_method : str, optional
        The unwrapping initialization method, by default "mcf"
    looks : Tuple[int, int], optional
        The number of looks in range and azimuth, by default (5, 1).
    alt_line_data : bool, optional
        Whether to use alternate line data, by default True.

    Returns
    -------
    subprocess.CompletedProcess
        The subprocess.CompletedProcess object from the unwrapping command.
        This object contains `.return_code`, `.stdout`, and `.stderr`

    Raises
    ------
    ValueError
        If the init_method is not "mcf" or "mst".
    CalledProcessError
        If the snaphu unwrapping command fails for some reason.
    """
    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    conncomp_file = Path(out_file).with_suffix(".unw.conncomp")
    cmd = _snaphu_cmd(
        fspath(ifg_file),
        fspath(cor_file or ""),
        Path(out_file),
        conncomp_file,
        fspath(mask_file or ""),
        do_tile=do_tile,
        alt_line_data=alt_line_data,
        init_method=init_method,
        looks=looks,
    )
    logger.debug(cmd)
    # copy_projection
    output = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    _save_with_metadata(
        ifg_file, out_file, alt_line_data=alt_line_data, dtype="float32"
    )
    _save_with_metadata(ifg_file, conncomp_file, alt_line_data=False, dtype="uint8")
    _set_unw_zeros(out_file, ifg_file, alt_line_data=alt_line_data)
    return output


def _snaphu_cmd(
    ifg_file,
    cor_file,
    out_file,
    conncomp_file,
    mask_file,
    do_tile=False,
    alt_line_data=True,
    init_method="mcf",
    looks=(5, 1),
):
    conf_name = out_file.with_suffix(out_file.suffix + ".snaphu_conf")
    width, _ = get_raster_xysize(ifg_file)
    # Need to specify the conncomp file format in a config file
    conf_string = f"""STATCOSTMODE SMOOTH
INFILE {ifg_file}
LINELENGTH {width}
OUTFILE {out_file}
CONNCOMPFILE {conncomp_file}   # TODO: snaphu has a bug for tiling conncomps
"""
    if alt_line_data:
        conf_string += "OUTFILEFORMAT		ALT_LINE_DATA\n"
    else:
        conf_string += "OUTFILEFORMAT		FLOAT_DATA\n"

    # Need to specify the input file format in a config file
    # the rest of the options are overwritten by command line options
    # conf_string += "INFILEFORMAT     COMPLEX_DATA\n"
    if cor_file:
        conf_string += "CORRFILEFORMAT   FLOAT_DATA\n"
        conf_string += f"CORRFILE	{cor_file}\n"

    if mask_file:
        conf_string += f"BYTEMASKFILE {mask_file}\n"

    # Calculate the tiles sizes/number of processes to use, separate for width/height
    nprocs = 1
    if do_tile:
        if width > 1000:
            conf_string += "NTILECOL 3\nCOLOVRLP 400\n"
            nprocs *= 3
            # cmd += " -S --tile 3 3 400 400 --nproc 9"
        elif width > 500:
            conf_string += "NTILECOL 2\nCOLOVRLP 400\n"
            nprocs *= 2
            # cmd += " -S --tile 2 2 400 400 --nproc 4"

        height = os.path.getsize(ifg_file) / width / 8
        if height > 1000:
            conf_string += "NTILEROW 3\nROWOVRLP 400\n"
            nprocs *= 3
        elif height > 500:
            conf_string += "NTILEROW 2\nROWOVRLP 400\n"
            nprocs *= 2
    if nprocs > 1:
        conf_string += f"NPROC {nprocs}\n"

    conf_string += f"INITMETHOD {init_method.upper()}\n"

    row_looks, col_looks = looks
    conf_string += "\n"
    conf_string += f"NLOOKSRANGE {col_looks}\n"
    conf_string += f"NLOOKSAZ {row_looks}\n"
    conf_string += f"NCORRLOOKS {row_looks * col_looks}\n"

    with open(conf_name, "w") as f:
        f.write(conf_string)

    cmd = f"snaphu -f {conf_name} "
    return cmd


def _set_unw_zeros(unw_filename, ifg_filename, alt_line_data=True):
    """Set areas that are 0 in the ifg to be 0 in the unw."""
    tmp_file = str(unw_filename).replace(".unw", "_tmp.unw")
    driver = "ENVI" if not alt_line_data else "ROI_PAC"
    cmd = (
        f"gdal_calc.py --quiet --outfile={tmp_file} --type=Float32 --co SUFFIX=ADD "
        f" --format={driver} --NoDataValue 0 --allBands=A -A {unw_filename} -B"
        f' {ifg_filename} --calc "A * (B!=0)"'
    )
    logger.debug(f"Setting zeros for {unw_filename}")
    logger.debug(cmd)
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(f"mv {tmp_file} {unw_filename}", shell=True)
    # remove the header file
    subprocess.check_call(f"rm -f {tmp_file}.*", shell=True)


def _save_with_metadata(like_file, raw_data_file, alt_line_data=True, dtype="float32"):
    """Write out a metadata file for `raw_data_file` using the `like_file` metadata."""
    cols, rows = get_raster_xysize(like_file)

    # Write a bare minimum auxiliary file so we can use gdal
    if alt_line_data:
        min_rsc = f"WIDTH {cols}\nFILE_LENGTH {rows}\n"
        with open(str(raw_data_file) + ".rsc", "w") as f:
            f.write(min_rsc)
    else:
        # Get ENVI data number l3harrisgeospatial.com/docs/enviheaderfiles.html
        if np.dtype(dtype) == np.float32:
            dt = 4
        elif np.dtype(dtype) == np.uint8:
            dt = 1
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
        min_hdr = (
            f"ENVI\nsamples = {cols}\nlines = {rows}\nbands = 1\ndata type = {dt}\n"
        )
        with open(str(raw_data_file) + ".hdr", "w") as f:
            f.write(min_hdr)

    copy_projection(like_file, raw_data_file)


@log_runtime
def run(
    ifg_path: Filename,
    output_path: Filename,
    cor_file: Optional[Filename] = "tcorr_ps_ds.bin",
    mask_file: Optional[Filename] = None,
    max_jobs: int = 10,
    overwrite: bool = False,
    no_tile: bool = True,
    init_method: str = "mcf",
):
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_path : Filename
        Path to input interferograms
    output_path : Filename
        Path to output directory
    cor_file : str, optional
        location of temporal correlation, by default "tcorr_ps_ds.bin"
    mask_file : Filename, optional
        Path to mask file, by default None
    max_jobs : int, optional
        Maximum parallel processes, by default 20
    overwrite : bool, optional
        overwrite results, by default False
    no_tile : bool, optional
        don't perform tiling on big interferograms, by default True
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mcf"

    Returns
    -------
    list[Path]
        list of unwrapped files names
    """
    filenames = list(Path(ifg_path).glob("*.int"))
    if len(filenames) == 0:
        logger.error("No files found. Exiting.")
        return

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    output_path = Path(output_path)

    ext_unw = ".unw"
    all_out_files = [(output_path / f.name).with_suffix(ext_unw) for f in filenames]
    in_files, out_files = [], []
    for inf, outf in zip(filenames, all_out_files):
        if os.path.exists(outf) and not overwrite:
            logger.info(f"{outf} exists. Skipping.")
            continue

        in_files.append(inf)
        out_files.append(outf)
    logger.info(f"{len(out_files)} left to unwrap")

    if mask_file:
        mask_file = Path(mask_file).resolve()

    with ThreadPoolExecutor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                inf,
                outf,
                cor_file,
                mask_file,
                not no_tile,
                init_method,
            )
            for inf, outf in zip(in_files, out_files)
        ]
        with progress:
            for fut in progress.track(as_completed(futures)):
                fut.result()

    return all_out_files
