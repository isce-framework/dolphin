"""Estimate wrapped phase from the DS in a stack of SLCS."""
import os
from os import fspath
from pathlib import Path

from atlas.utils import Pathlike, copy_projection

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
