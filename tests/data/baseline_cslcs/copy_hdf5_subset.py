#!/usr/bin/env python
import sys

import h5py


def create_hdf5_orbit_shell(source_file, dest_file):
    """Make a dummy OPERA CSLC file with data for baseline computation."""
    with h5py.File(source_file, "r") as src, h5py.File(dest_file, "w") as dst:
        # Copy the entire /metadata/orbit group
        orbit_group_dest = dst.require_group("/metadata/")
        data_group = dst.require_group("/data")
        wvl_group = dst.require_group(
            "/metadata/processing_information/input_burst_metadata"
        )
        src.copy("/metadata/orbit", orbit_group_dest)
        src.copy(
            "/metadata/processing_information/input_burst_metadata/wavelength",
            wvl_group,
        )

        datasets_to_copy = [
            "/data/x_coordinates",
            "/data/y_coordinates",
            "/data/projection",
        ]
        for dataset in datasets_to_copy:
            src.copy(dataset, data_group)


if __name__ == "__main__":
    try:
        src, dst = sys.argv[1:3]
    except IndexError:
        print(f"Usage: {sys.argv[0]} <src file> <dest file>")
        sys.exit(1)

    create_hdf5_orbit_shell(src, dst)
