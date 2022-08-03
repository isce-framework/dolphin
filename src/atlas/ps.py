"""Module for getting the persistent scatterers in a stack of SLCS."""
from osgeo_utils import gdal_calc

# ampdispersion.py -i stack/slcs_base.vrt -o ampDispersion/ampdispersion \
#   -m ampDispersion/mean
# gdal_calc.py --calc="a<0.42"  -a=ampDispersion/ampdispersion \
#   --outfile ampDispersion/ps_pixels --format ENVI --type Byte --overwrite


def create_amp_dispersion(
    *, input_vrt_file, output_file, amp_mean_file, reference_band, processing_opts
):
    """Create the amplitude dispersion file using FRInGE."""
    import ampdispersionlib

    aa = ampdispersionlib.Ampdispersion()

    aa.inputDS = input_vrt_file
    aa.outputDS = output_file
    aa.meanampDS = amp_mean_file

    aa.blocksize = processing_opts["lines_per_block"]
    aa.memsize = processing_opts["ram"]
    aa.refband = reference_band

    aa.run()


def create_ps(*, outfile, amp_disp_file, amp_dispersion_threshold: float = 0.42):
    """Create the PS file using the existing amplitude dispersion file."""
    gdal_calc.Calc(
        [f"a<{amp_dispersion_threshold}"],
        a=amp_disp_file,
        outfile=outfile,
        format="ENVI",
        type="Byte",
        overwrite=True,
        quiet=True,
    )
