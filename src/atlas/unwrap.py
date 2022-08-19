#!/usr/bin/env python
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import fspath
from pathlib import Path

from osgeo import gdal
from tqdm import tqdm

from atlas.log import get_log, log_runtime
from atlas.utils import Pathlike

logger = get_log()


def unwrap(
    intfile: Pathlike,
    corfile: Pathlike,
    outfile: Pathlike,
    width: int,
    do_tile: bool = False,
):
    """Unwrap a single interferogram."""
    conncomp_name = Path(intfile).with_suffix(".conncomp")
    cmd = _snaphu_cmd(
        intfile,
        corfile,
        outfile,
        conncomp_name,
        do_tile=do_tile,
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)
    # if os.path.exists(intfile + ".rsc"):
    # shutil.copy(intfile + ".rsc", outfile + ".rsc")
    # TODO: Need to make it gdal readable first
    # _set_unw_zeros(outfile, intfile)


def _snaphu_cmd(intfile, corname, outname, conncomp_name, do_tile=False):
    conf_name = outname.with_suffix(outname.suffix + ".snaphu_conf")
    width = _get_width(intfile)
    # Need to specify the conncomp file format in a config file
    conf_string = f"""STATCOSTMODE SMOOTH
INFILE {intfile}
LINELENGTH {width}
OUTFILE {outname}
CONNCOMPFILE {conncomp_name} # TODO: snaphu has a bug for tiling conncomps
"""
    # Need to specify the input file format in a config file
    # the rest of the options are overwritten by command line options
    # conf_string += "INFILEFORMAT     COMPLEX_DATA\n"
    # conf_string += "CORRFILEFORMAT   ALT_LINE_DATA"
    conf_string += "CORRFILEFORMAT   FLOAT_DATA\n"
    conf_string += f"CORRFILE	{corname}\n"

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

        height = os.path.getsize(intfile) / width / 8
        if height > 1000:
            conf_string += "NTILEROW 3\nROWOVRLP 400\n"
            nprocs *= 3
        elif height > 500:
            conf_string += "NTILEROW 2\nROWOVRLP 400\n"
            nprocs *= 2
    if nprocs > 1:
        conf_string += f"NPROC {nprocs}\n"

    with open(conf_name, "w") as f:
        f.write(conf_string)

    cmd = f"snaphu -f {conf_name} "
    return cmd


def _get_width(intfile):
    """Get the width of the interferogram."""
    ds = gdal.Open(fspath(intfile))
    width = ds.RasterXSize
    ds = None
    return width


def _set_unw_zeros(unw_filename, ifg_filename):
    """Set areas that are 0 in the ifg to be 0 in the unw."""
    tmp_file = unw_filename.replace(".unw", "_tmp.unw")
    cmd = (
        f"gdal_calc.py --quiet --outfile={tmp_file} --type=Float32 --format=ROI_PAC "
        f'--allBands=A -A {unw_filename} -B {ifg_filename} --calc "A * (B!=0)"'
    )
    print(f"Setting zeros for {unw_filename}")
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(f"mv {tmp_file} {unw_filename}", shell=True)
    subprocess.check_call(f"rm -f {tmp_file}.rsc", shell=True)


@log_runtime
def run(
    ifg_path: Pathlike,
    output_path: Pathlike,
    corfile: Pathlike = "tcorr_ps_ds.bin",
    max_jobs: int = 20,
    overwrite: bool = False,
    no_tile: bool = True,
    create_isce_headers: bool = False,
):
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_path : _type_
        Path to input interferograms
    output_path : _type_
        Path to output directory
    corfile : str, optional
        location of temporal correlation, by default "tcorr_ps_ds.bin"
    max_jobs : int, optional
        Maximum parallel processes, by default 20
    overwrite : bool, optional
        overwrite results, by default False
    no_tile : bool, optional
        don't perform tiling on big interferograms, by default True
    create_isce_headers : bool, optional
        Create .xml files for isce, by default False
    """
    filenames = list(Path(ifg_path).glob("*.int"))
    if len(filenames) == 0:
        logger.error("No files found. Exiting.")
        return

    output_path = Path(output_path)

    ds = gdal.Open(fspath(filenames[0]))
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    shape = (rows, cols)
    ds = None

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

    with ThreadPoolExecutor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                inf,
                corfile,
                outf,
                not no_tile,
            )
            for inf, outf in zip(in_files, out_files)
        ]
        for idx, fut in enumerate(tqdm(as_completed(futures)), start=1):
            fut.result()
            tqdm.write("Done with {} / {}".format(idx, len(futures)))

    if not create_isce_headers:
        return

    from apertools import isce_helpers, utils

    for f in tqdm(filenames):
        f = f.with_suffix(ext_unw)

        dirname, fname = os.path.split(f)
        with utils.chdir_then_revert(dirname):
            isce_helpers.create_unw_image(fname, shape=shape)
            # isce_helpers.create_int_image(fname)
