"""Combine estimated DS phases with PS phases to form interferograms."""
from pathlib import Path
from typing import List, Tuple

# import numpy as np
from osgeo import gdal

from atlas.utils import Pathlike, get_dates


def run_combine(
    *,
    input_vrt_file: Pathlike,
    ps_file: Pathlike,
    pl_directory: Pathlike,
    temp_coh_file: Pathlike,
    temp_coh_ps_ds_file: Pathlike,
    output_folder: Pathlike,
    ps_temp_coh: float,
):
    """Run workflow step to combine PS and DS phases."""
    # Create the temporal coherence file, filling in high values for PS
    fill_temp_coh(ps_file, temp_coh_file, temp_coh_ps_ds_file, ps_temp_coh)

    # Create the interferogram list
    pl_slc_files = Path(pl_directory).glob("*.slc")
    pl_date_dict = _make_date_file_dict(pl_slc_files)

    ds_orig_slc = gdal.Open(input_vrt_file)
    # The first file will be `stack_vrt_file`, the rest are the bands
    orig_slc_files = gdal.Info(ds_orig_slc, format="json")["files"][1:]
    orig_date_dict = _make_date_file_dict(orig_slc_files)
    assert len(pl_date_dict) == len(orig_date_dict)

    date_list = list(pl_date_dict.keys())
    date12_list = _make_ifg_list(date_list, "single-reference")

    xsize, ysize = ds_orig_slc.RasterXSize, ds_orig_slc.RasterYSize

    ds_psfile = gdal.Open(ps_file)
    bnd_ps = ds_psfile.GetRasterBand(1)

    driver = gdal.GetDriverByName("ENVI")
    for date_1, date_2 in date12_list:
        # output file witth both PS and DS pixels
        output_file = Path(output_folder) / f"{date_1}_{date_2}.int"
        # dataset for the output PS-DS integrated wrapped phase
        ds_out = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_CFloat32)
        bnd_out = ds_out.GetRasterBand(1)

        # get the current two SLCs, both original and phase-linked
        pl_file_1 = pl_date_dict[date_1]
        pl_file_2 = pl_date_dict[date_2]
        bnd_pl_1 = gdal.Open(pl_directory / pl_file_1).GetRasterBand(1)
        bnd_pl_2 = gdal.Open(pl_directory / pl_file_2).GetRasterBand(1)

        orig_file_1 = orig_date_dict[date_1]
        orig_file_2 = orig_date_dict[date_2]
        bnd_orig_1 = gdal.Open(orig_file_1).GetRasterBand(1)
        bnd_orig_2 = gdal.Open(orig_file_2).GetRasterBand(1)

        # integrate PS to DS for this pair and write to file block by block
        xsize, ysize = ds_out.RasterXSize, ds_out.RasterYSize
        x0, y0, xwindow, ywindow = _get_block_window(xsize, ysize, max_lines=1000)
        while y0 < ysize:
            cur_ywin = ywindow if (y0 + ywindow) < ysize else ysize - y0
            ps_arr = bnd_ps.ReadAsArray(x0, y0, xwindow, cur_ywin)

            ifg_out = _form_ifg(bnd_pl_1, bnd_pl_2, x0, y0, xwindow, ywindow)
            # Form the original full-res ifg
            ifg_orig = _form_ifg(bnd_orig_1, bnd_orig_2, x0, y0, xwindow, ywindow)
            # But only take the values at thhe PS pixels
            ps_mask = ps_arr == 1
            ifg_out[ps_mask] = ifg_orig[ps_mask]
            bnd_out.WriteArray(ifg_out, x0, y0)

            y0 += ywindow

        # close the datasets
        ds_orig_slc = None
        ds_out = bnd_out = None
        bnd_pl_1 = bnd_pl_2 = None
        bnd_orig_1 = bnd_orig_2 = None


def _form_ifg(bnd_1, bnd_2, x0, y0, xwindow, ywindow):
    # crossmultiply two SLCs
    # TODO: need to use actual crossmul module to avoid aliasing
    slc_1 = bnd_1.ReadAsArray(x0, y0, xwindow, ywindow)
    slc_2 = bnd_2.ReadAsArray(x0, y0, xwindow, ywindow)
    return slc_1 * slc_2.conj()
    # # This will normalize to 1 amplitude:
    # return np.exp(1j * np.angle(slc_2 * np.conjugate(slc_i)))


def _get_stack_file_list(stack_vrt_file):
    """Read in the dates and files contained in the bands of the VRT file."""
    ds = gdal.Open(stack_vrt_file)
    # The first file will be `stack_vrt_file`, the rest are the bands
    file_list = gdal.Info(ds, format="json")["files"][1:]
    ds = None
    return file_list


def _make_date_file_dict(file_list):
    file_paths = [Path(f) for f in file_list]
    dates = [get_dates(p)[0] for p in file_paths]
    # mapping from date to filename
    # filename_only = [str(f.name) for f in file_paths]
    # return {d: f for d, f in zip(dates, filename_only)}
    return {d: f for d, f in zip(dates, file_paths)}


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
    ds_in = gdal.Open(str(temp_coh_file))
    driver = gdal.GetDriverByName("ENVI")
    ds_out = driver.CreateCopy(str(temp_coh_ps_ds_file), ds_in)
    ds_in = None

    ds_psfile = gdal.Open(str(ps_file))
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


def _make_ifg_list(
    date_list: List[str], how="single-reference"
) -> List[Tuple[str, str]]:
    """Create a list of interferogram names from a list of dates."""
    if how == "single-reference":
        return _single_reference_network(date_list)
    else:
        raise NotImplementedError(f"{how} network not implemented")


def _single_reference_network(date_list: List[str]) -> List[Tuple[str, str]]:
    """Create a list of interferogram names from a list of dates."""
    ref = date_list[0]
    # return [f"{ref}_{date}" for date in date_list[1:]]
    return [(ref, date) for date in date_list[1:]]
