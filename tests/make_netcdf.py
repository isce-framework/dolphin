from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pyproj
from osgeo import gdal


def _add_complex_type(h5_root_group):
    if "complex64" in h5_root_group:
        return
    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(h5_root_group.id, np.bytes_("complex64"))


def create_test_nc(
    outfile,
    epsg=32615,
    subdir: Union[str, list[str]] = "/",
    data=None,
    data_ds_name="data",
    shape=(21, 15),
    dtype=np.complex64,
    xoff=0,
    yoff=0,
    write_mode="w",
    dx=1.0,
    dy=1.0,
):
    if isinstance(subdir, list):
        # Create groups in the same file to make multiple SubDatasets
        [
            create_test_nc(
                outfile, epsg, s, data, data_ds_name, shape, dtype, xoff, yoff, "a"
            )
            for s in subdir
        ]
        return

    if data is None:
        data = np.ones(shape, dtype=dtype)
    else:
        dtype = data.dtype
        shape = data.shape
    assert shape == data.shape, "Shape mismatch"
    assert data.dtype == dtype, "Data type mismatch"

    rows, cols = shape
    # Create basic HDF5 file
    hf = h5py.File(outfile, write_mode)
    hf.attrs["Conventions"] = "CF-1.8"

    xds = hf.create_dataset(
        os.path.join(subdir, "x"), data=xoff + (np.arange(cols) - cols / 2) * dx
    )
    yds = hf.create_dataset(
        os.path.join(subdir, "y"), data=yoff + (np.arange(rows, 0, -1) - rows / 2) * dy
    )

    if dtype == np.complex64:
        _add_complex_type(hf)

    datads = hf.create_dataset(os.path.join(subdir, data_ds_name), data=data)

    #  Mapping of dimension scales to datasets is not done automatically in HDF5
    #  We should label appropriate arrays as scales and attach them to datasets
    #  explicitly as show below.
    xds.make_scale()
    yds.make_scale()
    datads.dims[0].attach_scale(yds)
    datads.dims[1].attach_scale(xds)

    # Associate grid mapping with data
    srs_name = "spatial_ref"
    datads.attrs["grid_mapping"] = srs_name

    # Create a new single int dataset for projections
    srs_ds = hf.create_dataset(os.path.join(subdir, srs_name), (), dtype=int)
    srs_ds[()] = epsg

    # Set up pyproj for wkt
    crs = pyproj.CRS.from_epsg(epsg)
    # CF 1.7+ requires this attribute to be named "crs_wkt"
    # spatial_ref is old GDAL way. Using that for testing only.
    srs_ds.attrs[srs_name] = crs.to_wkt()
    srs_ds.attrs["crs_wkt"] = crs.to_wkt()

    srs_ds.attrs.update(crs.to_cf())

    xattrs, yattrs = crs.cs_to_cf()
    xds.attrs.update(xattrs)
    yds.attrs.update(yattrs)

    # Wrap up
    hf.close()


def get_cli_args():
    """Command line parser."""

    parser = argparse.ArgumentParser(description="CF tester")
    parser.add_argument(
        "-o",
        dest="outfile",
        type=str,
        required=True,
        help="Output file in CF conventions",
    )
    parser.add_argument(
        "-e", dest="epsg", type=int, default=4326, help="EPSG code to test"
    )
    parser.add_argument(
        "-r",
        dest="subdir",
        type=str,
        nargs="*",
        default="/",
        help=(
            "Group name(s). Root ('/') by default. If passing multiple "
            "groups, they will be created in the same file as subdatasets."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Create dummy NetCDF files containing CF-convention metadata."""

    # Command line parsing
    args = get_cli_args()

    # Check output extension is .nc (Only for verification with GDAL)
    if Path(args.outfile).suffix != ".nc":
        msg = (
            "This script uses GDAL's netcdf4 driver for verification and expects "
            "output file to have an extension of .nc"
        )
        raise ValueError(msg)

    create_test_nc(args.outfile, epsg=args.epsg, subdir=args.subdir)
    gdalinfo = gdal.Info(args.outfile, format="json")

    if len(args.subdir) == 1:
        assert f'ID["EPSG",{args.epsg}]]' in gdalinfo["coordinateSystem"]["wkt"]
    else:
        for i in range(1, len(args.subdir) + 1):
            gdalinfo_sub = gdal.Info(
                gdalinfo["metadata"]["SUBDATASETS"][f"SUBDATASET_{i}_NAME"],
                format="json",
            )

            assert f'ID["EPSG",{args.epsg}]]' in gdalinfo_sub["coordinateSystem"]["wkt"]
