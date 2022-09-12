"""Find the persistent scatterers in a stack of SLCS."""
from os import fspath

from dolphin.utils import Pathlike, copy_projection


def create_amp_dispersion(
    *,
    slc_vrt_file: Pathlike,
    output_file: Pathlike,
    amp_mean_file: Pathlike,
    reference_band: int,
    lines_per_block: int = 1000,
    ram: int = 1024,
):
    """Create the amplitude dispersion file using FRInGE."""
    import ampdispersionlib

    aa = ampdispersionlib.Ampdispersion()

    aa.inputDS = fspath(slc_vrt_file)
    aa.outputDS = fspath(output_file)
    aa.meanampDS = fspath(amp_mean_file)

    aa.blocksize = lines_per_block
    aa.memsize = ram
    aa.refband = reference_band

    aa.run()
    copy_projection(slc_vrt_file, output_file)


def create_ps(
    *,
    output_file: Pathlike,
    amp_disp_file: Pathlike,
    amp_dispersion_threshold: float = 0.42,
):
    """Create the PS file using the existing amplitude dispersion file."""
    from osgeo_utils import gdal_calc

    gdal_calc.Calc(
        [f"a<{amp_dispersion_threshold}"],
        a=fspath(amp_disp_file),
        outfile=fspath(output_file),
        format="ENVI",
        type="Byte",
        overwrite=True,
        quiet=True,
    )
    copy_projection(amp_disp_file, output_file)
