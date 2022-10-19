#!/usr/bin/env python
import itertools
import os
import re
import subprocess
from copy import copy
from pathlib import Path

# import apertools.colors  # for cmap='dismph'
import isce3
import numpy as np
import s1reader
import shapely
from osgeo import gdal

from dolphin import utils

gdal.UseExceptions()


def form_ifg_isce3(
    ref_filename,
    sec_filename,
    output_filename,
    row_looks=1,
    col_looks=1,
):
    print(f"using {ref_filename}, {sec_filename}")
    for fn in [ref_filename, sec_filename]:
        ds = gdal.Open(str(fn), gdal.GA_Update)
        ds.GetRasterBand(1).SetNoDataValue(np.nan)
        ds = None
    ref_slc_raster = isce3.io.Raster(str(ref_filename))
    sec_slc_raster = isce3.io.Raster(str(sec_filename))
    width = ref_slc_raster.width
    length = ref_slc_raster.length

    ifg_ext = utils.full_suffix(output_filename)

    if ifg_ext == ".tif":
        driver = "GTiff"
    elif ifg_ext == ".int":
        driver = "ISCE"
    else:
        driver = "ENVI"
    igram = isce3.io.Raster(
        str(output_filename),
        width // col_looks,
        length // row_looks,
        1,
        gdal.GDT_CFloat32,
        driver,
    )
    coherence_filename = Path(str(output_filename).replace(ifg_ext, ".cor.tif"))
    coherence = isce3.io.Raster(
        str(coherence_filename),
        width // col_looks,
        length // row_looks,
        1,
        gdal.GDT_Float32,
        "GTiff",
    )

    print(f"Running crossmul on input sized ({length = }, {width = })")
    print(
        f"Forming {output_filename}, {coherence_filename} with {row_looks = },"
        f" {col_looks = }"
    )
    # print(f"{cm.az_looks = }, {cm.range_looks = }, {cm.lines_per_block = }")
    cm = isce3.signal.Crossmul(col_looks, row_looks)
    cm.crossmul(ref_slc_raster, sec_slc_raster, igram, coherence)

    igram.close_dataset()
    coherence.close_dataset()
    # TODO: this may not work for the polar stereo...
    copy_projection_gdal(
        str(ref_filename), str(output_filename), ylooks=row_looks, xlooks=col_looks
    )
    copy_projection_gdal(
        str(ref_filename), str(coherence_filename), ylooks=row_looks, xlooks=col_looks
    )


def make_all_ifgs_stitched(looks=(10, 20), slc_dir="stitched", ifg_subdir="ifgs"):
    slc_files = list(sorted(Path(f"{slc_dir}/").glob("*.slc")))

    ifg_dir = Path(f"{slc_dir}/{ifg_subdir}/")
    ifg_dir.mkdir(exist_ok=True)
    out_names = []
    for f1, f2 in itertools.combinations(slc_files, 2):
        date1 = re.search(r"\d{8}", str(f1)).group()
        date2 = re.search(r"\d{8}", str(f2)).group()
        out_name = ifg_dir / f"{date1}_{date2}.int.tif"
        form_ifg_isce3(f1, f2, out_name, *looks)
        out_names.append(out_name)
    return out_names


def make_all_burst_ifgs(looks=(10, 20), slc_dir="stack", ifg_subdir="ifgs"):
    burst_dirs = list(sorted(Path(f"{slc_dir}/").glob("t*_iw*")))
    out_names = []
    for bd in burst_dirs:
        slc_files = list(sorted(bd.rglob("*/*.slc")))
        ifg_dir = bd / ifg_subdir
        ifg_dir.mkdir(exist_ok=True)
        print(bd, ifg_dir)
        for f1, f2 in itertools.combinations(slc_files, 2):
            date1 = re.search(r"\d{8}", str(f1)).group()
            date2 = re.search(r"\d{8}", str(f2)).group()
            out_name = ifg_dir / f"{date1}_{date2}.int.tif"
            print(f"Forming {out_name}")
            form_ifg_isce3(f1, f2, out_name, *looks)
            out_names.append(out_name)
        print(10 * "_")
    return out_names


def merge_burst_ifgs(slc_dir="stack", ifg_subdir="ifgs", merged_ifg_dir="merged_ifgs"):
    # Make output directory
    Path(f"{slc_dir}/{merged_ifg_dir}").mkdir(exist_ok=True)
    # Get the dates of all SLCs
    dates = [p.name for p in sorted(next(Path(slc_dir).glob("t*iw*")).glob("./2*"))]

    for d1, d2 in itertools.combinations(dates, 2):
        ifg_date = f"{d1}_{d2}"
        for ext in [".int.tif", ".cor.tif"]:
            output = f"{slc_dir}/{merged_ifg_dir}/{ifg_date}{ext}"
            inputs = f"{slc_dir}/*/{ifg_subdir}/{ifg_date}{ext}"
            cmd = f"gdal_merge.py -init nan -n nan -o {output} {inputs}"
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)


def copy_projection_gdal(src_file, dst_file, ylooks=1, xlooks=1) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    ds_src = gdal.Open(os.fspath(src_file))
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        print("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(os.fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        gt_looked = copy(list(geotransform))
        gt_looked[1] *= xlooks
        gt_looked[5] *= ylooks
        ds_dst.SetGeoTransform(gt_looked)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def main():
    # # Download the data of interest: get one month of data in the northeast
    # # installed from apertools to get `asfdownload`
    dp = Path("data/")
    if not dp.exists() or not dp.glob("*.zip"):
        subprocess.check_call(
            "asfdownload --start 2022-01-01 --end 2022-02-01 --wkt aoi.wkt --out-dir"
            " data",
            shell=True,
        )

    fnames = list(sorted(dp.glob("*.zip")))
    Path("orbits").mkdir(exist_ok=True)

    orbit_files = list(sorted(Path("orbits").glob("*EOF")))
    if not orbit_files or len(orbit_files) < len(fnames):
        orbit_files = [
            s1reader.s1_orbit.download_orbit(fname, "orbits") for fname in fnames
        ]

    if not Path("elevation.tif").exists():
        burst_lists = [
            s1reader.load_bursts(fnames[0], orbit_files[0], swath_num=i)
            for i in [1, 2, 3]
        ]

        all_borders = [b.border[0] for b in itertools.chain.from_iterable(burst_lists)]
        bbox_frame = shapely.ops.unary_union(all_borders).bounds

        bbox_str = " ".join((str(s) for s in bbox_frame))
        cmd = (
            f"sardem --bbox {bbox_str} --data-source COP -o elevation.tif -of GTiff -ot"
            " Float32"
        )
        # TODO: Do i actually just want the AOI to be the bounds? aoi.wkt
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    subprocess.check_call(
        "/home/staniewi/repos/COMPASS/src/compass/s1_geo_stack.py --slc-dir data/"
        " --dem-file elevation.tif --orbit-dir ./orbits/ --aoi aoi.wkt ",
        shell=True,
    )
    # Run all the generated files to geocode
    subprocess.check_call(
        "for f in stack/run_files/*sh; do bash $f >> process.log 2>&1; done", shell=True
    )

    # ## Stitch all bursts together

    subprocess.check_call(
        "/home/staniewi/repos/COMPASS/src/compass/utils/stitching/stitch_burst.py"
        " --burst-dir stack/ -o stitched",
        shell=True,
    )

    # Form all possible interferogram pairs
    make_all_ifgs_stitched(looks=(10, 20))


if __name__ == "__main__":
    main()
    # TODO: get path, get number looks, save plot
    # Get date
