#!/usr/bin/env python

import argparse
import os

import h5py
import numpy as np
import pyproj
from osgeo import gdal


def _add_complex_type(h5_root_group):
    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(h5_root_group.id, np.string_("complex64"))


def create_test_nc(
    outfile, epsg=32615, subdir="/", data=None, shape=(21, 15), dtype=np.complex64
):
    if data is None:
        data = np.ones(shape, dtype=dtype)
    else:
        dtype = data.dtype
        shape = data.shape
    assert shape == data.shape, "Shape mismatch"
    assert data.dtype == dtype, "Data type mismatch"

    rows, cols = shape
    # Create basic HDF5 file
    hf = h5py.File(outfile, "w")
    hf.attrs["Conventions"] = "CF-1.8"

    xds = hf.create_dataset(
        os.path.join(subdir, "x"), data=(np.arange(cols) - cols / 2)
    )
    yds = hf.create_dataset(
        os.path.join(subdir, "y"), data=np.arange(rows, 0, -1) - rows / 2
    )

    if dtype == np.complex64:
        _add_complex_type(hf)

    datads = hf.create_dataset(os.path.join(subdir, "data"), data=data)

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
        "-r", dest="subdir", type=str, default="/", help="Group name. Root by default"
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Driver for testing CF."""

    # Command line parsing
    args = get_cli_args()

    # Check output extension is .nc (Only for verification with GDAL)
    if os.path.splitext(args.outfile)[1] != ".nc":
        raise Exception(
            "This script uses GDAL's netcdf4 driver for verification and expects output"
            " file to have an extension of .nc"
        )

    create_test_nc(args.outfile, epsg=args.epsg, subdir=args.subdir)
    gdalinfo = gdal.Info(args.outfile, format="json")
    print(gdalinfo["coordinateSystem"]["wkt"])
