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


def update_amp_disp(
    amp_mean_file: Pathlike,
    amp_disp_file: Pathlike,
    slc_vrt_file: Pathlike,
):
    """Update the amplitude dispersion for the new SLC.

    Uses Welford's method to update the mean and variance.

    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm  # noqa: E501
    or https://changyaochen.github.io/welford/ for derivation.

    mu_{n+1} = mu_n + (x_{n+1} - mu_n) / (n+1)
    var_{n+1} = var_n + ((x_{n+1} - mu_n) * (x_{n+1} - mu_{n+1}) - var_n) / (n+1)

    v1 = v0 + (x1 - m0) * (x1 - m1)

    References
    ----------
    Welford, B. P. "Note on a method for calculating corrected sums of squares and
    products." Technometrics 4.3 (1962): 419-420.
    """
    import numpy as np
    from osgeo import gdal

    # N = int(gdal.Info(fspath(amp_mean_file), format="json")["metadata"]["N"])
    ds_mean = gdal.Open(fspath(amp_mean_file), gdal.GA_Update)
    try:
        # Get the number of SLCs used to create the mean amplitude
        N = int(ds_mean.GetMetadataItem("N"))
    except KeyError:
        ds_mean = None
        raise ValueError("Cannot find N in metadata of mean amplitude file")

    bnd_mean = ds_mean.GetRasterBand(1)
    mean_n = bnd_mean.ReadAsArray()

    ds_ampdisp = gdal.Open(fspath(amp_disp_file), gdal.GA_Update)
    bnd_ampdisp = ds_ampdisp.GetRasterBand(1)
    ampdisp = bnd_ampdisp.ReadAsArray()

    # Get the new data amplitude
    ds_slc_stack = gdal.Open(fspath(slc_vrt_file))
    nbands = ds_slc_stack.RasterCount
    # The last band should be the new SLC
    bnd_new_slc = ds_slc_stack.GetRasterBand(nbands)
    new_amp = np.abs(bnd_new_slc.ReadAsArray())
    bnd_new_slc = ds_slc_stack = None

    # Get the variance from the amplitude dispersion
    # d = sigma / mu, so sigma^2 = d^2 * mu^2
    var_n = ampdisp**2 * mean_n**2

    # Update the mean
    mean_n1 = mean_n + (new_amp - mean_n) / (N + 1)
    # Update the variance
    var_n1 = var_n + ((new_amp - mean_n) * (new_amp - mean_n1) - var_n) / (N + 1)

    # Update both files with the new values
    bnd_mean.WriteArray(mean_n1)
    bnd_ampdisp.WriteArray(np.sqrt(var_n1 / mean_n1**2))

    # Update the metadata with the new N
    ds_ampdisp.SetMetadataItem("N", str(N + 1))
    ds_mean.SetMetadataItem("N", str(N + 1))

    # Close the files to save the changes
    bnd_mean = bnd_ampdisp = ds_mean = ds_ampdisp = None
