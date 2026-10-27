from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from dolphin import io
from dolphin.stack import CompressedSlcInfo, MiniStackInfo, MiniStackPlanner
from dolphin.utils import flatten

NUM_ACQ = 10


# Note: uses the fixtures from conftest.py
# Use a smaller subset for these tests
@pytest.fixture()
def files(slc_file_list):
    return slc_file_list[:NUM_ACQ]


@pytest.fixture()
def files_nc(slc_file_list_nc):
    return slc_file_list_nc[:NUM_ACQ]


@pytest.fixture()
def dates(slc_date_list):
    return slc_date_list[:NUM_ACQ]


@pytest.fixture()
def is_compressed(files):
    return [False] * len(files[:NUM_ACQ])


@pytest.fixture()
def date_lists(dates):
    # To mimic what we get back from running `get_dates` on a list of files
    return [[d] for d in dates]


def test_create_ministack(tmp_path, files, date_lists, is_compressed):
    MiniStackInfo(file_list=files, is_compressed=is_compressed, dates=date_lists)

    # Check it works by providing a real local dir
    MiniStackInfo(
        file_list=files,
        dates=date_lists,
        is_compressed=is_compressed,
        output_folder=tmp_path,
    )

    # Check it works when making up a future output dir which doesn't exist
    MiniStackInfo(
        file_list=files,
        dates=date_lists,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
    )


def test_mismatched_lengths(files, date_lists, is_compressed):
    with pytest.raises(ValueError):
        MiniStackInfo(
            file_list=files, dates=date_lists[1:], is_compressed=is_compressed
        )

    with pytest.raises(ValueError):
        MiniStackInfo(
            file_list=files, dates=date_lists, is_compressed=is_compressed[:-1]
        )


def test_create_ministack_dates(files, dates, is_compressed):
    # Check we can provide a list of dates instead of tuples
    m = MiniStackInfo(file_list=files, dates=dates, is_compressed=is_compressed)
    assert m.dates[0] == [
        dates[0],
    ]


@pytest.fixture()
def ministack(files, date_lists, is_compressed):
    return MiniStackInfo(
        file_list=files,
        dates=date_lists,
        is_compressed=is_compressed,
        output_folder="fake_dir",
    )


def test_ministack_attrs(ministack):
    assert ministack.real_slc_date_range_str == "20220101_20220110"


def test_create_compressed_slc(files, date_lists, is_compressed):
    m = MiniStackInfo(file_list=files, is_compressed=is_compressed, dates=date_lists)
    comp_slc = m.get_compressed_slc_info()
    flat_dates = list(flatten(date_lists))
    assert comp_slc.real_slc_dates == flat_dates
    assert comp_slc.real_slc_file_list == files

    # Now try one where we marked the first of the stack as compressed
    is_compressed2 = is_compressed.copy()
    is_compressed2[0] = True
    m2 = MiniStackInfo(file_list=files, is_compressed=is_compressed2, dates=date_lists)
    comp_slc2 = m2.get_compressed_slc_info()
    assert comp_slc2.real_slc_dates == flat_dates[1:]
    assert comp_slc2.real_slc_file_list == files[1:]
    assert comp_slc2.compressed_slc_file_list == files[0:1]


def run_ministack_planner(files, date_lists, is_compressed):
    msp = MiniStackPlanner(
        file_list=files,
        dates=date_lists,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
        max_num_compressed=5,
    )
    ms_list = msp.plan(4)
    assert len(ms_list) == 3

    ms_size = 3
    ms_list = msp.plan(ms_size)
    assert len(ms_list) == 4

    expected_out_folders = [
        Path("fake_dir/20220101_20220103"),
        Path("fake_dir/20220104_20220106"),
        Path("fake_dir/20220107_20220109"),
        Path("fake_dir/20220110_20220110"),
    ]
    for idx, ms in enumerate(ms_list):
        # size should increment by 1 each time
        if idx < 3:
            assert len(ms.file_list) == idx + 3
        else:
            # Last one is 1 real, + 3 compressed
            assert len(ms.file_list) == 4

        # number of compressed should increment by 1 each time
        assert sum(ms.is_compressed) == idx
        assert ms.file_list[idx:] == files[ms_size * idx : ms_size * (1 + idx)]

        assert ms.output_folder == expected_out_folders[idx]

    assert all(ms.compressed_reference_date == datetime(2022, 1, 1) for ms in ms_list)
    assert all(ms.output_reference_date == datetime(2022, 1, 1) for ms in ms_list)


def test_ministack_planner_gtiff(files, date_lists, is_compressed):
    run_ministack_planner(files, date_lists, is_compressed)


# Unclear how to parameterize over the 2 fixtures
def test_ministack_planner_nc(files_nc, date_lists, is_compressed):
    run_ministack_planner(files_nc, date_lists, is_compressed)


@pytest.fixture()
def ministack_planner(files, date_lists, is_compressed):
    return MiniStackPlanner(
        file_list=files,
        dates=date_lists,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
        max_num_compressed=5,
    )


def test_ccslc_infos(ministack_planner):
    # Check we can get the compressed SLC info
    ministacks = ministack_planner.plan(3)
    flat_dates = list(flatten(ministack_planner.dates))
    m = ministacks[0]
    ccslc = m.get_compressed_slc_info()
    assert ccslc.real_slc_dates == flat_dates[0:3]
    assert ccslc.real_slc_file_list == ministack_planner.file_list[0:3]
    assert ccslc.compressed_slc_file_list == []

    m2 = ministacks[1]
    ccslc2 = m2.get_compressed_slc_info()
    assert ccslc2.real_slc_dates == flat_dates[3:6]
    assert ccslc2.real_slc_file_list == ministack_planner.file_list[3:6]
    assert ccslc2.compressed_slc_file_list == [ccslc.path]


def test_ccslc_round_trip_metadata(ministack_planner, tmp_path):
    ministacks = ministack_planner.plan(3)
    m = ministacks[1]
    m.output_folder = tmp_path
    ccslc = m.get_compressed_slc_info()

    io.write_arr(arr=np.ones((2, 2)), output_name=ccslc.path)
    ccslc.write_metadata()

    md_dict = io.get_raster_metadata(ccslc.path, domain="DOLPHIN")
    assert set(md_dict.keys()) == ccslc.model_dump().keys()

    c2 = CompressedSlcInfo.from_file_metadata(ccslc.path)
    assert c2.real_slc_dates == ccslc.real_slc_dates
    # files will have turned to strings in the dump
    assert list(map(str, ccslc.real_slc_file_list)) == c2.real_slc_file_list


def test_ccslc_moved(ministack_planner, tmp_path):
    ministacks = ministack_planner.plan(3)
    m = ministacks[1]
    m.output_folder = tmp_path
    ccslc = m.get_compressed_slc_info()
    io.write_arr(arr=np.ones((2, 2)), output_name=ccslc.path)
    ccslc.write_metadata()

    other_path = tmp_path / "other"
    other_path.mkdir()
    new_name = other_path / ccslc.path.name
    ccslc.path.rename(new_name)

    c2 = CompressedSlcInfo.from_file_metadata(new_name)
    assert c2.real_slc_dates == ccslc.real_slc_dates
    assert list(map(str, ccslc.real_slc_file_list)) == c2.real_slc_file_list
    # Only the `output_folder` should have changed
    assert c2.output_folder == other_path


def test_hit_max_compressed(slc_file_list, slc_date_list, is_compressed):
    """Check the planning works even after going pased the max CCSLCs."""
    is_compressed = [False] * len(slc_file_list)
    ministack_planner = MiniStackPlanner(
        file_list=slc_file_list,
        dates=slc_date_list,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
        max_num_compressed=3,
    )
    # Check we can get the compressed SLC info
    ministacks = ministack_planner.plan(3)
    assert len(ministacks[0].file_list) == 3
    assert len(ministacks[1].file_list) == 4
    assert len(ministacks[2].file_list) == 5
    assert len(ministacks[3].file_list) == 6
    assert all(len(m.file_list) == 6 for m in ministacks[3:-1])


def test_compressed_idx_setting(slc_file_list, slc_date_list, is_compressed):
    """Check the planning works to set a manually passed compressed index."""
    is_compressed = [False] * len(slc_file_list)
    ministack_planner = MiniStackPlanner(
        file_list=slc_file_list,
        dates=slc_date_list,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
    )
    # Check we can get the compressed SLC info
    expected_ref_date = slc_date_list[8]

    # This currently works ONLY for one single ministack planning
    # (e.g. manually separating ministack runs and re-passing in CCSLCs)
    ministacks = ministack_planner.plan(len(slc_file_list), compressed_idx=8)

    assert ministacks[0].compressed_reference_date == expected_ref_date
    # But, the output reference date should be the first one (since it was unset)
    assert ministacks[0].output_reference_date == slc_date_list[0]


def test_compressed_plans_always_first(slc_file_list, slc_date_list, is_compressed):
    """Check the planning works to set a manually passed compressed index."""
    is_compressed = [False] * len(slc_file_list)
    ministack_planner = MiniStackPlanner(
        file_list=slc_file_list,
        dates=slc_date_list,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
        compressed_slc_plan="always_first",
    )

    # This currently works ONLY for one single ministack planning
    # (e.g. manually separating ministack runs and re-passing in CCSLCs)
    ms_size = 3
    ministacks = ministack_planner.plan(ms_size)

    compslcs = [m.get_compressed_slc_info() for m in ministacks]
    assert all(c.reference_date == slc_date_list[0] for c in compslcs)


def test_compressed_plans_last_per_ministack(
    slc_file_list, slc_date_list, is_compressed
):
    """Check the planning works to set a manually passed compressed index."""
    is_compressed = [False] * len(slc_file_list)
    # This currently works ONLY for one single ministack planning
    with pytest.raises(ValueError):
        ministack_planner = MiniStackPlanner(
            file_list=slc_file_list,
            dates=slc_date_list,
            is_compressed=is_compressed,
            output_folder=Path("fake_dir"),
            compressed_slc_plan="last_per_ministack",
        )
        ministacks = ministack_planner.plan(3)

    ms_size = 4
    ministack_planner = MiniStackPlanner(
        file_list=slc_file_list[:ms_size],
        dates=slc_date_list[:ms_size],
        is_compressed=is_compressed[:ms_size],
        output_folder=Path("fake_dir"),
        compressed_slc_plan="last_per_ministack",
    )
    ministacks = ministack_planner.plan(ms_size)
    compslcs = [m.get_compressed_slc_info() for m in ministacks]
    assert len(compslcs) == 1
    assert compslcs[0].reference_date == slc_date_list[ms_size - 1]
