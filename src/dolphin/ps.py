"""Find the persistent scatterers in a stack of SLCS."""
from os import fspath
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm.auto import tqdm

gdal.UseExceptions()

from dolphin._types import Filename
from dolphin.io import save_arr, save_block


def create_ps(
    # *,
    slc_vrt_file: Filename,
    output_file: Filename,
    amp_mean_file: Filename,
    amp_dispersion_file: Filename,
    amp_dispersion_threshold: float = 0.42,
    max_ram_gb: float = 1.0,
):
    """Create the amplitude dispersion, mean, and PS files.

    Parameters
    ----------
    slc_vrt_file : Filename
        The VRT file pointing to the stack of SLCs.
    output_file : Filename
        The output PS file (dtype: Byte)
    amp_dispersion_file : Filename
        The output amplitude dispersion file.
    amp_mean_file : Filename
        The output mean amplitude file.
    amp_dispersion_threshold : float, optional
        The threshold for the amplitude dispersion. Default is 0.42.
    max_ram_gb : int, optional
        The maximum amount of data to read at a time (in GB).
        Default is 1.0 GB.
    """
    from .vrt import VRTStack

    # Initialize the output files with zeros
    types = [np.uint8, np.float32, np.float32]
    file_list = [output_file, amp_dispersion_file, amp_mean_file]
    nodatas = [255, 0, 0]
    for fn, dtype, nodata in zip(file_list, types, nodatas):
        save_arr(
            arr=None,
            like_filename=slc_vrt_file,
            output_name=fn,
            nbands=1,
            dtype=dtype,
            nodata=nodata,
        )

    vrt_stack = VRTStack.from_vrt_file(slc_vrt_file)
    max_bytes = 1e9 * max_ram_gb
    num_blocks = vrt_stack._get_num_blocks(max_bytes=max_bytes)
    block_shape = vrt_stack._get_block_shape(max_bytes=max_bytes)

    # Initialize the intermediate arrays for the calculation
    magnitude = np.zeros((len(vrt_stack), *block_shape), dtype=np.float32)

    # Make the generator for the blocks
    block_gen = vrt_stack.iter_blocks(
        return_slices=True,
        max_bytes=max_bytes,
        skip_empty=False,
    )
    for cur_data, (rows, cols) in tqdm(block_gen, total=num_blocks):
        if np.all(cur_data == 0) or np.all(np.isnan(cur_data)):
            continue

        cur_rows, cur_cols = cur_data.shape[-2:]
        magnitude_cur = np.abs(cur_data, out=magnitude[:cur_rows, cur_cols])
        mean, amp_disp, ps = calc_ps_block(magnitude_cur, amp_dispersion_threshold)

        # Use the UInt8 type for the PS to save.
        # For invalid pixels, set to max Byte value
        ps = ps.astype(np.uint8)
        ps[amp_disp == 0] = 255

        # Write amp dispersion and the mean blocks
        save_block(mean, amp_mean_file, rows, cols)
        save_block(amp_disp, amp_dispersion_file, rows, cols)
        save_block(ps, output_file, rows, cols)


def calc_ps_block(stack_mag: np.ndarray, amp_dispersion_threshold: float = 0.42):
    """Calculate the amplitude dispersion for a block of data.

    Parameters
    ----------
    stack_mag : np.ndarray
        The magnitude of the stack of SLCs.
    amp_dispersion_threshold : float, optional
        The threshold for the amplitude dispersion to label a pixel as a PS:
            ps = amp_disp < amp_dispersion_threshold
        Default is 0.42.

    Returns
    -------
    mean : np.ndarray
        The mean amplitude for the block.
        dtype: float32
    amp_disp : np.ndarray
        The amplitude dispersion for the block.
    ps : np.ndarray
        The persistent scatterers for the block.
        dtype: bool
    """
    # Make the nans into 0s to ignore them
    np.nan_to_num(stack_mag, copy=False)

    # TODO: is it worth creating each ndarray in advance and use `out=`?
    mean = np.nanmean(stack_mag, axis=0)
    std_dev = np.nanstd(stack_mag, axis=0)

    # Calculate the amplitude dispersion and replace nans with 0s
    amp_disp = mean / std_dev
    amp_disp = np.nan_to_num(amp_disp, nan=0, posinf=0, neginf=0, copy=False)

    ps = amp_disp < amp_dispersion_threshold
    ps[amp_disp == 0] = False
    return mean, amp_disp, ps


def update_amp_disp(
    amp_mean_file: Filename,
    amp_dispersion_file: Filename,
    slc_vrt_file: Filename,
    output_directory: Filename = "",
):
    r"""Update the amplitude dispersion for a new SLC.

    Uses Welford's method for online updating of mean and variance.

    \[
    \begin{align}
    \mu_{n+1} &= \mu_n + (x_{n+1} - \mu_n) / (n+1)  \\
    \text{var}_{n+1} &= \text{var}_n + ((x_{n+1} - \mu_n) * (x_{n+1} - \mu_{n+1}) - \text{var}_n) / (n+1) \\
    v1 &= v0 + (x1 - m0) * (x1 - m1)
    \end{align}
    \]


    See <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>
    or <https://changyaochen.github.io/welford/> for derivation.


    Parameters
    ----------
    amp_mean_file : Filename
        The existing mean amplitude file.
    amp_dispersion_file : Filename
        The existing amplitude dispersion file.
    slc_vrt_file : Filename
        The VRT file pointing to the stack of SLCs.
        Assumes that the final band is the new SLC to be added.
    output_directory : Filename, optional
        The output directory for the updated files, current directory by default.

    References
    ----------
    Welford, B. P. "Note on a method for calculating corrected sums of squares and
    products." Technometrics 4.3 (1962): 419-420.
    """  # noqa: E501
    output_directory = Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    output_mean_file = output_directory / Path(amp_mean_file).name
    output_disp_file = output_directory / Path(amp_dispersion_file).name

    _check_output_files(output_mean_file, output_disp_file)

    ds_mean = gdal.Open(fspath(amp_mean_file), gdal.GA_ReadOnly)
    ds_ampdisp = gdal.Open(fspath(amp_dispersion_file), gdal.GA_ReadOnly)
    # Get the number of SLCs used to create the mean amplitude
    try:
        # Use the ENVI metadata domain for ENVI files
        md_domain = "ENVI" if ds_mean.GetDriver().ShortName == "ENVI" else ""
        N = int(ds_mean.GetMetadataItem("N", md_domain))
    except KeyError:
        ds_mean = ds_ampdisp = None  # Close files before raising error
        raise ValueError("Cannot find N in metadata of mean amplitude file")

    driver = ds_mean.GetDriver()
    mean_n = ds_mean.GetRasterBand(1).ReadAsArray()
    ampdisp = ds_ampdisp.GetRasterBand(1).ReadAsArray()

    # Get the new data amplitude
    ds_slc_stack = gdal.Open(fspath(slc_vrt_file))
    nbands = ds_slc_stack.RasterCount
    # The last band should be the new SLC
    bnd_new_slc = ds_slc_stack.GetRasterBand(nbands)
    new_amp = np.abs(bnd_new_slc.ReadAsArray())
    bnd_new_slc = ds_slc_stack = None

    # Make the output files
    ds_mean_out = driver.CreateCopy(fspath(output_mean_file), ds_mean)
    ds_ampdisp_out = driver.CreateCopy(fspath(output_disp_file), ds_ampdisp)
    bnd_mean = ds_mean_out.GetRasterBand(1)
    bnd_ampdisp = ds_ampdisp_out.GetRasterBand(1)

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
    ds_ampdisp.SetMetadataItem("N", str(N + 1), md_domain)
    ds_mean.SetMetadataItem("N", str(N + 1), md_domain)

    # Close the files to save the changes
    bnd_mean = bnd_ampdisp = ds_mean = ds_ampdisp = None
    ds_mean = ds_ampdisp = None


def _check_output_files(*files):
    """Check if the output files already exist."""
    err_msg = "Output file {} already exists. Please delete before running."
    for f in files:
        if f.exists():
            raise FileExistsError(err_msg.format(f))
