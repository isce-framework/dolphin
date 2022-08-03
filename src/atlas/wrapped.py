"""Module for estimating wrapped phase from the DS in a stack of SLCS."""
import os

# nmap.py -i stack/slcs_base.vrt -o nmap/nmap -c nmap/count -x 11 -y 5


def run_nmap(
    *,
    input_vrt_file,
    weight_file,
    nmap_count_file,
    window,
    pvalue,
    stat_method,
    no_gpu,
    processing_opts,
    mask_file=None,
):
    """Find the SHP neighborhoods of pixels in the stack of SLCs using FRInGE."""
    import nmaplib

    aa = nmaplib.Nmap()

    aa.inputDS = input_vrt_file
    aa.weightsDS = weight_file
    aa.countDS = nmap_count_file
    if mask_file:
        aa.maskDS = mask_file

    aa.halfWindowX = window["xhalf"]
    aa.halfWindowY = window["yhalf"]

    aa.minimumProbability = pvalue
    aa.method = stat_method
    aa.noGPU = no_gpu

    aa.blocksize = processing_opts["lines_per_block"]
    aa.memsize = processing_opts["ram"]

    aa.run()


# time evd.py -i stack/slcs_base.vrt -o EVD -x 11 -y 5 -w nmap/nmap


def run_evd(args):
    """Run the EVD algorithm on a stack of SLCs using FRInGE."""
    import evdlib

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    aa = evdlib.Evd()

    # Explicit wiring. Can be automated later.
    aa.inputDS = args.input_ds
    aa.weightsDS = args.weights_ds

    aa.outputFolder = aa.outputCompressedSlcFolder = args.output_folder
    aa.compSlc = args.compressed_slc_file  # "compslc.bin"

    aa.blocksize = args.linesperblock
    aa.memsize = args.memorySize
    aa.halfWindowX = args.halfWindowX
    aa.halfWindowY = args.halfWindowY
    aa.minimumNeighbors = args.min_neighbors

    # Set up method and bandwidth
    aa.method = args.method

    aa.run()
