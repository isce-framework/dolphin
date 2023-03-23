#!/usr/bin/env python
import argparse

import h5py
import numpy as np

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log()


class ComparisonError(Exception):
    """Exception raised when two datasets do not match."""

    pass


def compare_groups(golden_group: h5py.Group, test_group: h5py.Group) -> None:
    """Compare all datasets in two HDF5 files.

    Parameters
    ----------
    golden_group : h5py.Group
        Path to the golden file.
    test_group : h5py.Group
        Path to the test file to be compared.

    Raises
    ------
    ComparisonError
        If the two files do not match in all datasets.
    """
    # Check if group names match
    if set(golden_group.keys()) != set(test_group.keys()):
        raise ComparisonError(
            f"Group keys do not match: {set(golden_group.keys())} vs"
            f" {set(test_group.keys())}"
        )

    for key in golden_group.keys():
        if isinstance(golden_group[key], h5py.Group):
            compare_groups(golden_group[key], test_group[key])
        else:
            _compare_datasets_attr(golden_group[key], test_group[key])
            np.testing.assert_array_almost_equal(
                golden_group[key][()], test_group[key][()], decimal=5
            )


def _compare_datasets_attr(
    golden_dataset: h5py.Dataset, test_dataset: h5py.Dataset
) -> None:
    if golden_dataset.shape != test_dataset.shape:
        raise ComparisonError(
            f"Dataset shapes do not match: {golden_dataset.shape} vs"
            f" {test_dataset.shape}"
        )

    if golden_dataset.dtype != test_dataset.dtype:
        raise ComparisonError(
            f"Dataset dtypes do not match: {golden_dataset.dtype} vs"
            f" {test_dataset.dtype}"
        )

    if golden_dataset.name != test_dataset.name:
        raise ComparisonError(
            f"Dataset names do not match: {golden_dataset.name} vs {test_dataset.name}"
        )

    if golden_dataset.attrs.keys() != test_dataset.attrs.keys():
        raise ComparisonError(
            f"Dataset attribute keys do not match: {golden_dataset.attrs.keys()} vs"
            f" {test_dataset.attrs.keys()}"
        )

    for attr_key in golden_dataset.attrs.keys():
        if attr_key in ("REFERENCE_LIST", "DIMENSION_LIST"):
            continue
        val1, val2 = golden_dataset.attrs[attr_key], test_dataset.attrs[attr_key]
        if isinstance(val1, np.ndarray):
            is_equal = np.allclose(val1, val2, equal_nan=True)
        elif isinstance(val1, np.floating) and np.isnan(val1) and np.isnan(val2):
            is_equal = True
        else:
            is_equal = val1 == val2
        if not is_equal:
            raise ComparisonError(
                f"Dataset attribute values for key '{attr_key}' do not match: "
                f"{golden_dataset.attrs[attr_key]} vs {test_dataset.attrs[attr_key]}"
            )


def check_raster_geometadata(golden_file: Filename, test_file: Filename) -> None:
    """Check if the raster metadata (bounds, CRS, and GT) match.

    Parameters
    ----------
    golden_file : Filename
        Path to the golden file.
    test_file : Filename
        Path to the test file to be compared.

    Raises
    ------
    ComparisonError
        If the two files do not match in their metadata
    """
    funcs = [io.get_raster_bounds, io.get_raster_crs, io.get_raster_gt]
    for func in funcs:
        val_golden = func(golden_file)
        val_test = func(test_file)
        if val_golden != val_test:
            raise ComparisonError(f"{func} does not match: {val_golden} vs {val_test}")


def main() -> None:
    """Compare two HDF5 files for consistency."""
    parser = argparse.ArgumentParser(
        description="Compare two HDF5 files for consistency."
    )
    parser.add_argument("golden", help="The golden HDF5 file.")
    parser.add_argument("test", help="The test HDF5 file to be compared.")
    parser.add_argument(
        "--data-dset", default="/science/SENTINEL1/DISP/unwrapped_phase"
    )
    args = parser.parse_args()

    try:
        logger.info("Comparing HDF5 files...")
        with h5py.File(args.golden, "r") as golden, h5py.File(args.test, "r") as test:
            compare_groups(golden, test)
        check_raster_geometadata(
            io.format_nc_filename(args.golden, args.data_dset),
            io.format_nc_filename(args.test, args.data_dset),
        )
    except Exception as e:
        raise ComparisonError(
            f"Unexpected error comparing {args.golden} and {args.test}."
        ) from e

    logger.info(f"Files {args.golden} and {args.test} match.")


if __name__ == "__main__":
    main()
