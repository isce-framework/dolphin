"""Estimate wrapped phase from the DS in a stack of SLCS."""
import os

# nmap.py -i stack/slcs_base.vrt -o nmap/nmap -c nmap/count -x 11 -y 5


def run_nmap(
    *,
    slc_vrt_file,
    weight_file,
    nmap_count_file,
    window,
    nmap_opts,
    mask_file=None,
    lines_per_block=128,
    ram=1024,
    no_gpu=False,
):
    """Find the SHP neighborhoods of pixels in the stack of SLCs using FRInGE."""
    import nmaplib

    aa = nmaplib.Nmap()

    aa.inputDS = slc_vrt_file
    aa.weightsDS = weight_file
    aa.countDS = nmap_count_file
    if mask_file:
        aa.maskDS = mask_file

    aa.halfWindowX = window["xhalf"]
    aa.halfWindowY = window["yhalf"]

    aa.minimumProbability = nmap_opts["pvalue"]
    aa.method = nmap_opts["stat_method"]

    aa.blocksize = lines_per_block
    aa.memsize = ram
    aa.noGPU = no_gpu

    aa.run()


def run_evd(
    *,
    slc_vrt_file,
    weight_file,
    compressed_slc_filename,
    output_folder,
    window,
    pl_opts,
    lines_per_block=128,
    ram=1024,
):
    """Run the EVD algorithm on a stack of SLCs using FRInGE."""
    import evdlib

    # TODO: Check into how the multiple EVD threads are spawned
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    aa = evdlib.Evd()

    # Explicit wiring. Can be automated later.
    aa.inputDS = slc_vrt_file
    aa.weightsDS = weight_file

    aa.outputFolder = aa.outputCompressedSlcFolder = output_folder
    aa.compSlc = compressed_slc_filename  # "compslc.bin"

    aa.minimumNeighbors = pl_opts["minimum_neighbors"]
    aa.method = pl_opts["method"].upper()

    aa.halfWindowX = window["xhalf"]
    aa.halfWindowY = window["yhalf"]

    aa.blocksize = lines_per_block
    aa.memsize = ram

    aa.run()
