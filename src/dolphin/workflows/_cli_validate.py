#!/usr/bin/env python
import argparse
from typing import TYPE_CHECKING, Any, Optional

import h5py
import numpy as np

from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename

logger = get_log()

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any

DSET_DEFAULT = "/science/SENTINEL1/DISP/unwrapped_phase"


class ComparisonError(Exception):
    """Exception raised when two datasets do not match."""

    pass


def compare_groups(
    golden_group: h5py.Group,
    test_group: h5py.Group,
    pixels_failed_threshold: float = 0.01,
    diff_threshold: float = 1e-5,
) -> None:
    """Compare all datasets in two HDF5 files.

    Parameters
    ----------
    golden_group : h5py.Group
        Path to the golden file.
    test_group : h5py.Group
        Path to the test file to be compared.
    pixels_failed_threshold : float, optional
        The threshold of the percentage of pixels that can fail the comparison.
    diff_threshold : float, optional
        The abs. difference threshold between pixels to consider failing.

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
            img_gold = np.ma.masked_invalid(golden_group[key][()])
            img_test = np.ma.masked_invalid(test_group[key][()])
            abs_diff = np.abs((img_gold.filled(0) - img_test.filled(0)))
            num_failed = np.count_nonzero(abs_diff > diff_threshold)
            # num_pixels = np.count_nonzero(~np.isnan(img_gold))  # do i want this?
            num_pixels = img_gold.size
            if num_failed / num_pixels > pixels_failed_threshold:
                raise ComparisonError(
                    f"Dataset {golden_group.name}/{key} values do not match: Number of"
                    f" pixels failed: {num_failed} / {num_pixels} ="
                    f" {100*num_failed / num_pixels:.2f}%"
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


def _check_raster_geometadata(golden_file: Filename, test_file: Filename) -> None:
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
        val_golden = func(golden_file)  # type: ignore
        val_test = func(test_file)  # type: ignore
        if val_golden != val_test:
            raise ComparisonError(f"{func} does not match: {val_golden} vs {val_test}")


def compare(golden: Filename, test: Filename, data_dset: str = DSET_DEFAULT) -> None:
    """Compare two HDF5 files for consistency."""
    try:
        logger.info("Comparing HDF5 contents...")
        with h5py.File(golden, "r") as hf_g, h5py.File(test, "r") as hf_t:
            compare_groups(hf_g, hf_t)

        logger.info("Cheaking geospatial metadata...")
        _check_raster_geometadata(
            io.format_nc_filename(golden, data_dset),
            io.format_nc_filename(test, data_dset),
        )
    except ComparisonError:
        raise

    except Exception as e:
        raise ComparisonError(f"Unexpected error comparing {golden} and {test}.") from e

    logger.info(f"Files {golden} and {test} match.")


def get_parser(
    subparser: Optional[_SubparserType] = None, subcommand_name: str = "run"
) -> argparse.ArgumentParser:
    """Set up the command line interface."""
    metadata = dict(
        description="Compare two HDF5 files for consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)  # type: ignore
    else:
        parser = argparse.ArgumentParser(**metadata)  # type: ignore

    parser.add_argument("golden", help="The golden HDF5 file.")
    parser.add_argument("test", help="The test HDF5 file to be compared.")
    parser.add_argument("--data-dset", default=DSET_DEFAULT)
    parser.set_defaults(run_func=compare)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    compare(args.golden, args.test, args.data_dset)
