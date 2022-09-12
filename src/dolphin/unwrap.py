#!/usr/bin/env python
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import Optional

import numpy as np
from osgeo import gdal
from tqdm import tqdm

from dolphin.log import get_log, log_runtime
from dolphin.utils import Pathlike, numpy_to_gdal_type

logger = get_log()

gdal.UseExceptions()


def unwrap(
    ifg_file: Pathlike,
    cor_file: Pathlike,
    out_file: Pathlike,
    mask_file: Optional[Pathlike],
    do_tile: bool = False,
    init_method: str = "mcf",
):
    """Unwrap a single interferogram."""
    conncomp_file = Path(out_file).with_suffix(".unw.conncomp")
    alt_line_data = True
    tmp_intfile = _nan_to_zero(ifg_file)
    cmd = _snaphu_cmd(
        fspath(tmp_intfile),
        cor_file,
        out_file,
        conncomp_file,
        mask_file,
        do_tile=do_tile,
        alt_line_data=alt_line_data,
        init_method=init_method,
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)
    _save_with_metadata(
        tmp_intfile, out_file, alt_line_data=alt_line_data, dtype="float32"
    )
    _save_with_metadata(tmp_intfile, conncomp_file, alt_line_data=False, dtype="byte")
    _set_unw_zeros(out_file, tmp_intfile)
    os.remove(tmp_intfile)


def _nan_to_zero(infile):
    """Make a copy of infile and replace NaNs with 0."""
    in_p = Path(infile)
    tmp_file = (in_p.parent) / (in_p.stem + "_tmp" + in_p.suffix)

    ds_in = gdal.Open(fspath(infile))
    drv = ds_in.GetDriver()
    ds_out = drv.CreateCopy(fspath(tmp_file), ds_in)

    bnd = ds_in.GetRasterBand(1)
    nodata = bnd.GetNoDataValue()
    arr = bnd.ReadAsArray()
    mask = np.logical_or(np.isnan(arr), arr == nodata)
    arr[mask] = 0
    ds_out.GetRasterBand(1).WriteArray(arr)
    ds_out = None
    # cmd = (
    #     f"gdal_calc.py -A {infile} --out_file={tmp_file} --overwrite "
    #     "--NoDataValue 0 --format ROI_PAC --calc='np.isnan(A)*0 + (~np.isnan(A))*A' "
    # )
    # logger.info(cmd)
    # subprocess.check_call(cmd, shell=True)
    return tmp_file


def _snaphu_cmd(
    ifg_file,
    cor_file,
    out_file,
    conncomp_file,
    mask_file,
    do_tile=False,
    alt_line_data=True,
    init_method="mcf",
):
    conf_name = out_file.with_suffix(out_file.suffix + ".snaphu_conf")
    width = _get_width(ifg_file)
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
    # conf_string += "CORRFILEFORMAT   ALT_LINE_DATA"
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

    conf_string += f"INIT_METHOD {init_method.upper()}\n"

    with open(conf_name, "w") as f:
        f.write(conf_string)

    cmd = f"snaphu -f {conf_name} "
    return cmd


def _get_width(ifg_file):
    """Get the width of the interferogram."""
    ds = gdal.Open(fspath(ifg_file))
    width = ds.RasterXSize
    ds = None
    return width


def _set_unw_zeros(unw_filename, ifg_filename):
    """Set areas that are 0 in the ifg to be 0 in the unw."""
    tmp_file = str(unw_filename).replace(".unw", "_tmp.unw")
    cmd = (
        f"gdal_calc.py --quiet --out_file={tmp_file} --type=Float32 --format=ROI_PAC "
        f'--allBands=A -A {unw_filename} -B {ifg_filename} --calc "A * (B!=0)"'
    )
    print(f"Setting zeros for {unw_filename}")
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(f"mv {tmp_file} {unw_filename}", shell=True)
    subprocess.check_call(f"rm -f {tmp_file}.rsc", shell=True)


def _save_with_metadata(meta_file, data_file, alt_line_data=True, dtype="float32"):
    """Write out a metadata file for `data_file` using the `meta_file` metadata."""
    ds = gdal.Open(fspath(meta_file))
    rows = ds.RasterYSize
    cols = ds.RasterXSize

    # read in using fromfile, since we can't use gdal yet
    dtype = np.dtype(str(dtype).lower())
    data = np.fromfile(data_file, dtype=dtype)
    if alt_line_data:
        amp = data.reshape((rows, 2 * cols))[:, :cols]
        phase = data.reshape((rows, 2 * cols))[:, cols:]
        driver = "ROI_PAC"
        nbands = 2
    else:
        amp = None
        phase = data.reshape((rows, cols))
        driver = "ENVI"
        nbands = 1

    gdal_dt = numpy_to_gdal_type(dtype)
    drv = gdal.GetDriverByName(driver)
    options = ["SUFFIX=ADD"] if driver == "ENVI" else []
    ds_out = drv.Create(fspath(data_file), cols, rows, nbands, gdal_dt, options=options)
    # print("saving to", data_file, "with driver", driver)
    if amp is None:
        bnd = ds_out.GetRasterBand(1)
        bnd.WriteArray(phase)
    else:
        bnd = ds_out.GetRasterBand(1)
        bnd.WriteArray(amp)
        bnd = ds_out.GetRasterBand(2)
        bnd.WriteArray(phase)
    bnd = ds_out = None


@log_runtime
def run(
    ifg_path: Pathlike,
    output_path: Pathlike,
    cor_file: Pathlike = "tcorr_ps_ds.bin",
    mask_file: Pathlike = None,
    max_jobs: int = 20,
    overwrite: bool = False,
    no_tile: bool = True,
    create_isce_headers: bool = False,
    init_method: str = "mcf",
):
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_path : Pathlike
        Path to input interferograms
    output_path : Pathlike
        Path to output directory
    cor_file : str, optional
        location of temporal correlation, by default "tcorr_ps_ds.bin"
    mask_file : Pathlike, optional
        Path to mask file, by default None
    max_jobs : int, optional
        Maximum parallel processes, by default 20
    overwrite : bool, optional
        overwrite results, by default False
    no_tile : bool, optional
        don't perform tiling on big interferograms, by default True
    create_isce_headers : bool, optional
        Create .xml files for isce, by default False
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mcf"
    """
    filenames = list(Path(ifg_path).glob("*.int"))
    if len(filenames) == 0:
        logger.error("No files found. Exiting.")
        return

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

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

    if mask_file:
        mask_file = Path(mask_file).absolute()

    with ThreadPoolExecutor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                inf,
                cor_file,
                outf,
                mask_file,
                not no_tile,
                init_method,
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
