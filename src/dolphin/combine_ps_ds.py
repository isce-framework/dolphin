"""Combine estimated DS phases with PS phases to form interferograms."""
from os import fspath
from pathlib import Path

import numpy as np
from osgeo import gdal
from osgeo_utils import gdal_calc

from dolphin import io, network
from dolphin.log import get_log
from dolphin.utils import Pathlike, get_dates

gdal.UseExceptions()

logger = get_log()


def form_ifgs(
    *,
    slc_vrt_file: Pathlike,
    pl_directory: Pathlike,
    output_folder: Pathlike,
    ifg_network_options: dict,
    driver: str = "GTiff",
):
    """Run workflow step to combine PS and DS phases."""
    # Create the interferogram list
    pl_slc_files = Path(pl_directory).glob("*.slc.tif")
    pl_date_dict = {get_dates(p)[0]: p for p in pl_slc_files}

    ds_orig_stack = gdal.Open(fspath(slc_vrt_file))
    assert len(pl_date_dict) == ds_orig_stack.RasterCount

    date_list = [k for k in pl_date_dict.keys() if k]
    date12_list = network.make_ifg_list(date_list, **ifg_network_options)

    for date_1, date_2 in date12_list:
        # output file with both PS and DS pixels
        output_file = Path(output_folder) / f"{date_1}_{date_2}.int"
        slc1_path = pl_date_dict[date_1]
        slc2_path = pl_date_dict[date_2]
        logger.info(
            f"Forming interferogram {output_file.stem} in {output_folder} between"
            f" {slc1_path} and {slc2_path}"
        )

        gdal_calc.Calc(
            NoDataValue=np.nan,
            format=driver,
            outfile=output_file,
            A=fspath(slc1_path),
            B=fspath(slc2_path),
            calc="A * B.conj()",
            quiet=True,
            overwrite=True,
            creation_options=io.DEFAULT_TIFF_OPTIONS,
        )


def run_combine(
    *,
    slc_vrt_file: Pathlike,
    ps_file: Pathlike,
    pl_directory: Pathlike,
    temp_coh_file: Pathlike,
    temp_coh_ps_ds_file: Pathlike,
    output_folder: Pathlike,
    ps_temp_coh: float,
    ifg_network_options: dict,
):
    """Run workflow step to combine PS and DS phases."""
    # Create the interferogram list
    pl_slc_files = Path(pl_directory).glob("*.slc.tif")
    pl_date_dict = {get_dates(p)[0]: p for p in pl_slc_files}

    ds_orig_stack = gdal.Open(fspath(slc_vrt_file))
    assert len(pl_date_dict) == ds_orig_stack.RasterCount

    date_list = [k for k in pl_date_dict.keys() if k]
    date12_list = network.make_ifg_list(date_list, **ifg_network_options)

    ds_psfile = gdal.Open(fspath(ps_file))
    bnd_ps = ds_psfile.GetRasterBand(1)

    xsize, ysize = ds_orig_stack.RasterXSize, ds_orig_stack.RasterYSize
    driver = gdal.GetDriverByName("ENVI")
    for date_1, date_2 in date12_list:
        # output file with both PS and DS pixels
        output_file = Path(output_folder) / f"{date_1}_{date_2}.int"
        # dataset for the output PS-DS integrated wrapped phase
        ds_out = driver.Create(fspath(output_file), xsize, ysize, 1, gdal.GDT_CFloat32)
        bnd_out = ds_out.GetRasterBand(1)

        logger.info(f"Forming interferogram {output_file.stem} in {output_folder}")
        # get the current two SLCs, both original and phase-linked
        pl_file_1 = pl_date_dict[date_1]
        pl_file_2 = pl_date_dict[date_2]
        ds_pl_1 = gdal.Open(fspath(pl_file_1))
        ds_pl_2 = gdal.Open(fspath(pl_file_2))
        bnd_pl_1 = ds_pl_1.GetRasterBand(1)
        bnd_pl_2 = ds_pl_2.GetRasterBand(1)

        idx1 = date_list.index(date_1)
        idx2 = date_list.index(date_2)
        bnd_orig_1 = ds_orig_stack.GetRasterBand(idx1 + 1)
        bnd_orig_2 = ds_orig_stack.GetRasterBand(idx2 + 1)

        # integrate PS to DS for this pair and write to file block by block
        xsize, ysize = ds_out.RasterXSize, ds_out.RasterYSize
        x0, y0, xwindow, ywindow = _get_block_window(xsize, ysize, max_lines=1000)
        while y0 < ysize:
            # Limit the y window to remaining lines
            cur_ywin = ywindow if (y0 + ywindow) < ysize else ysize - y0
            ps_arr = bnd_ps.ReadAsArray(x0, y0, xwindow, cur_ywin)

            # Form the original full-res ifg
            ifg_orig = _form_ifg(bnd_orig_1, bnd_orig_2, x0, y0, xwindow, cur_ywin)
            # TODO do I want to keep the amplitude=1 ?
            ifg_out = _form_ifg(bnd_pl_1, bnd_pl_2, x0, y0, xwindow, cur_ywin)
            # Use the original ifg's amplitude values:
            ifg_out = np.abs(ifg_orig) * np.exp(1j * np.angle(ifg_out))

            # breakpoint()
            # But only take the values at the PS pixels
            ps_mask = ps_arr == 1
            # ps_mask = np.zeros_like(ifg_orig, dtype=bool)
            ifg_out[ps_mask] = ifg_orig[ps_mask]
            bnd_out.WriteArray(ifg_out, x0, y0)
            bnd_out.FlushCache()

            y0 += ywindow

        # close the datasets
        ds_out = bnd_out = None
        ds_pl_1 = ds_pl_2 = bnd_pl_1 = bnd_pl_2 = None
        # ds_orig_1 = ds_orig_2 = bnd_orig_1 = bnd_orig_2 = None
        bnd_orig_1 = bnd_orig_2 = None
    ds_orig_stack = None

    # Create the temporal coherence file, filling in high values for PS
    fill_temp_coh(ps_file, temp_coh_file, temp_coh_ps_ds_file, ps_temp_coh)
    return


def _form_ifg(bnd_1, bnd_2, x0, y0, xwindow, ywindow):
    # cross multiply two SLCs
    # TODO: need to use actual crossmul module to avoid aliasing
    slc_1 = bnd_1.ReadAsArray(x0, y0, xwindow, ywindow)
    slc_2 = bnd_2.ReadAsArray(x0, y0, xwindow, ywindow)
    return slc_1 * slc_2.conj()


def fill_temp_coh(
    ps_file: Pathlike,
    temp_coh_file: Pathlike,
    temp_coh_ps_ds_file: Pathlike,
    ps_temp_coh: float,
):
    """Fill in high values for PS in the temporal coherence file.

    Parameters
    ----------
    ps_file : Pathlike
        Name of persistent scatterer binary file.
    temp_coh_file : Pathlike
        Name of temporal coherence file resulting from phase linking.
    temp_coh_ps_ds_file : Pathlike
        Name of output temporal coherence file with PS values filled in.
    ps_temp_coh : float
        Value to fill in at the PS pixels in the merged temporal coherence file.
    """
    # Start with a copy of the PS temporal coherence file
    ds_in = gdal.Open(fspath(temp_coh_file))
    driver = gdal.GetDriverByName("ENVI")
    ds_out = driver.CreateCopy(fspath(temp_coh_ps_ds_file), ds_in)
    ds_in = None

    ds_psfile = gdal.Open(fspath(ps_file))
    bnd_ps = ds_psfile.GetRasterBand(1)
    bnd_out = ds_out.GetRasterBand(1)

    # Overwrite the temporal coherence at PS points with a high value, in blocks
    # Use the full width each time, iterate over the height dimension
    xsize, ysize = ds_out.RasterXSize, ds_out.RasterYSize
    x0, y0, xwindow, ywindow = _get_block_window(xsize, ysize, max_lines=1000)
    while y0 < ysize:
        cur_ywin = ywindow if (y0 + ywindow) < ysize else ysize - y0
        ps_arr = bnd_ps.ReadAsArray(x0, y0, xwindow, cur_ywin)
        tc_merged = bnd_out.ReadAsArray(x0, y0, xwindow, cur_ywin)

        ps_mask = ps_arr == 1
        tc_merged[ps_mask] = ps_temp_coh
        bnd_out.WriteArray(tc_merged, x0, y0)

        y0 += ywindow

    ds_out = bnd_out = ds_psfile = bnd_ps = None


def _get_block_window(xsize, ysize, max_lines=1000):
    x0, y0 = 0, 0
    xwindow = xsize
    ywindow = min(max_lines, ysize)
    return x0, y0, xwindow, ywindow
