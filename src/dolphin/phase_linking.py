"""Estimate wrapped phase from the DS in a stack of SLCS."""
import os
from os import fspath
from pathlib import Path

from dolphin.utils import Pathlike, copy_projection

# nmap.py -i stack/slcs_base.vrt -o nmap/nmap -c nmap/count -x 11 -y 5


def run_nmap(
    *,
    slc_vrt_file: Pathlike,
    weight_file: Pathlike,
    nmap_count_file: Pathlike,
    window: dict,
    nmap_opts: dict,
    mask_file: Pathlike = None,
    lines_per_block: int = 128,
    ram: int = 1024,
    no_gpu: bool = False,
):
    """Find the SHP neighborhoods of pixels in the stack of SLCs using FRInGE."""
    import nmaplib

    aa = nmaplib.Nmap()

    aa.inputDS = fspath(slc_vrt_file)
    aa.weightsDS = fspath(weight_file)
    aa.countDS = fspath(nmap_count_file)
    if mask_file:
        aa.maskDS = fspath(mask_file)

    aa.halfWindowX = window["xhalf"]
    aa.halfWindowY = window["yhalf"]

    aa.minimumProbability = nmap_opts["pvalue"]
    aa.method = nmap_opts["stat_method"]

    aa.blocksize = lines_per_block
    aa.memsize = ram
    aa.noGPU = no_gpu

    aa.run()

    copy_projection(slc_vrt_file, nmap_count_file)
    copy_projection(slc_vrt_file, weight_file)


def create_full_nmap_files(
    xsize,
    ysize,
    half_window_x,
    half_window_y,
    output_dir="nmap",
    nmap_filename="nmap",
    count_filename="count",
):
    """Make dummy SHP nmap/count files with all neighborhoods set to 1.

    Used to avoid running nmap if you just want to multi-look a window.
    """
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    nmap_file = output_dir / f"{nmap_filename}"
    count_file = output_dir / f"{count_filename}"

    wy = 1 + half_window_y * 2
    wx = 1 + half_window_x * 2

    drv = gdal.GetDriverByName("ENVI")
    options = ["INTERLEAVE=BIP", "SUFFIX=ADD"]
    # number of uint32 bytes needed to store weights
    n_bands = int(np.ceil(wx * wy / 32))
    ds_nmap = drv.Create(
        fspath(nmap_file), xsize, ysize, n_bands, gdal.GDT_UInt32, options
    )
    ds_count = drv.Create(fspath(count_file), xsize, ysize, 1, gdal.GDT_Int16, options)

    # set all nmap weights to 1 by have all bits set to 1.
    # 0xFFFFFFFF = 4294967295
    uint32_max = 2**32 - 1
    for b in range(1, n_bands + 1):
        ds_nmap.GetRasterBand(b).WriteArray(
            np.full((ysize, xsize), uint32_max, dtype=np.uint32)
        )

    # Make the counts full too
    count = np.full((ysize, xsize), wx * wy, dtype=np.int16)
    ds_count.GetRasterBand(1).WriteArray(count)

    for ds in [ds_nmap, ds_count]:
        ds.SetMetadata(
            {"HALFWINDOWX": str(half_window_x), "HALFWINDOWY": str(half_window_y)}
        )
    ds_nmap = ds_count = None


def run_evd(
    *,
    slc_vrt_file: Pathlike,
    weight_file: Pathlike,
    compressed_slc_file: Pathlike,
    output_folder: Pathlike,
    window: dict,
    pl_opts: dict,
    lines_per_block: int = 128,
    ram: int = 1024,
):
    """Run the EVD algorithm on a stack of SLCs using FRInGE."""
    import evdlib
    import phase_linklib

    # TODO: Check into how the multiple EVD threads are spawned
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    method = pl_opts["method"].upper()
    if method in ["EVD", "MLE"]:
        aa = evdlib.Evd()
    elif method in ("PHASE_LINK", "PL"):
        aa = phase_linklib.Phaselink()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Explicit wiring. Can be automated later.
    aa.inputDS = fspath(slc_vrt_file)
    aa.weightsDS = fspath(weight_file)

    aa.outputFolder = aa.outputCompressedSlcFolder = fspath(output_folder)
    aa.compSlc = fspath(Path(compressed_slc_file).name)  # "compslc.bin"

    aa.minimumNeighbors = pl_opts["minimum_neighbors"]
    aa.method = method

    aa.halfWindowX = window["xhalf"]
    aa.halfWindowY = window["yhalf"]

    aa.blocksize = lines_per_block
    aa.memsize = ram

    aa.run()
    copy_projection(slc_vrt_file, compressed_slc_file)
