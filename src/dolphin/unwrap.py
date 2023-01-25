import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import fspath
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal
from tqdm import tqdm

from dolphin._types import Filename
from dolphin.io import get_raster_xysize
from dolphin.log import get_log, log_runtime
from dolphin.utils import numpy_to_gdal_type

logger = get_log()

gdal.UseExceptions()


def unwrap(
    ifg_file: Filename,
    cor_file: Filename,
    out_file: Filename,
    mask_file: Optional[Filename],
    do_tile: bool = False,
    init_method: str = "mcf",
    looks: Tuple[int, int] = (5, 1),
):
    """Unwrap a single interferogram."""
    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")
    conncomp_file = Path(out_file).with_suffix(".unw.conncomp")
    alt_line_data = True
    cmd = _snaphu_cmd(
        fspath(ifg_file),
        fspath(cor_file),
        Path(out_file),
        conncomp_file,
        mask_file,
        do_tile=do_tile,
        alt_line_data=alt_line_data,
        init_method=init_method,
        looks=looks,
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)
    _save_with_metadata(
        ifg_file, out_file, alt_line_data=alt_line_data, dtype="float32"
    )
    _save_with_metadata(ifg_file, conncomp_file, alt_line_data=False, dtype="byte")
    _set_unw_zeros(out_file, ifg_file)


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


def _set_unw_zeros(unw_filename, ifg_filename):
    """Set areas that are 0 in the ifg to be 0 in the unw."""
    tmp_file = str(unw_filename).replace(".unw", "_tmp.unw")
    cmd = (
        f"gdal_calc.py --quiet --outfile={tmp_file} --type=Float32 --format=ROI_PAC "
        f'--allBands=A -A {unw_filename} -B {ifg_filename} --calc "A * (B!=0)"'
    )
    print(f"Setting zeros for {unw_filename}")
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(f"mv {tmp_file} {unw_filename}", shell=True)
    # remove the header file
    subprocess.check_call(f"rm -f {tmp_file}.*", shell=True)


def _save_with_metadata(meta_file, data_file, alt_line_data=True, dtype="float32"):
    """Write out a metadata file for `data_file` using the `meta_file` metadata."""
    cols, rows = get_raster_xysize(meta_file)

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
    ifg_path: Filename,
    output_path: Filename,
    cor_file: Filename = "tcorr_ps_ds.bin",
    mask_file: Optional[Filename] = None,
    max_jobs: int = 20,
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
    """
    filenames = list(Path(ifg_path).glob("*.int"))
    if len(filenames) == 0:
        logger.error("No files found. Exiting.")
        return

    if init_method.lower() not in ("mcf", "mst"):
        raise ValueError(f"Invalid init_method {init_method}")

    output_path = Path(output_path)

    cols, rows = get_raster_xysize(filenames[0])

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
