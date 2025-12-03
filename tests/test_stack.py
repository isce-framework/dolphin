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


def test_real_slc_indices(files, date_lists, is_compressed):
    # Test with all real SLCs
    m = MiniStackInfo(file_list=files, is_compressed=is_compressed, dates=date_lists)
    real_indices = m.real_slc_indices
    assert len(real_indices) == len(files)
    assert list(real_indices) == list(range(len(files)))
    compressed_indices = m.compressed_slc_indices
    assert len(compressed_indices) == 0

    # Test with some compressed SLCs
    is_compressed2 = is_compressed.copy()
    is_compressed2[0] = True
    is_compressed2[3] = True
    is_compressed2[7] = True
    m2 = MiniStackInfo(file_list=files, is_compressed=is_compressed2, dates=date_lists)
    real_indices2 = m2.real_slc_indices
    compressed_indices2 = m2.compressed_slc_indices
    assert list(real_indices2) == [1, 2, 4, 5, 6, 8, 9]
    assert list(compressed_indices2) == [0, 3, 7]
    assert len(real_indices2) + len(compressed_indices2) == len(files)


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
        # With the new logic, compressed SLCs in last 5 dates are removed
        # But only if at least 2 SLCs remain after removal
        if idx < 3:
            # Would be idx + 3, but compressed SLCs removed from last 5
            assert len(ms.file_list) == 3
            # Compressed SLCs removed, so count is 0
            assert sum(ms.is_compressed) == 0
            # Real SLC files should match the current ministack's real SLCs
            expected_real = files[ms_size * idx : ms_size * (1 + idx)]
            assert ms.file_list == expected_real
        else:
            # Last ministack: 3 compressed + 1 real = 4 total
            # Removal would leave only 1 SLC, so safety check prevents removal
            assert len(ms.file_list) == 4
            # 3 compressed SLCs remain
            assert sum(ms.is_compressed) == 3

        assert ms.output_folder == expected_out_folders[idx]

    # Reference dates changed with compressed SLC removal logic
    # Ministacks 0-2: no compressed SLCs, so reference is their own first date
    # Ministack 3: has compressed SLCs, so reference is the last compressed SLC
    expected_compressed_ref = [
        datetime(2022, 1, 1),  # ministack 0: first real SLC
        datetime(2022, 1, 4),  # ministack 1: first real SLC (no compressed)
        datetime(2022, 1, 7),  # ministack 2: first real SLC (no compressed)
        datetime(2022, 1, 7),  # ministack 3: last compressed SLC (from ms 2)
    ]
    expected_output_ref = expected_compressed_ref.copy()
    for idx, ms in enumerate(ms_list):
        assert ms.compressed_reference_date == expected_compressed_ref[idx]
        assert ms.output_reference_date == expected_output_ref[idx]


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
    # Compressed SLC from ministack[0] would be in last 5 dates, so it's removed
    assert ccslc2.compressed_slc_file_list == []


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
    # Note: Compressed SLCs in the last 5 dates are removed to avoid
    # redundancy in the interferogram network
    assert len(ministacks[0].file_list) == 3
    # ministacks[1] would have 4 (1 compressed + 3 real) but compressed is removed
    assert len(ministacks[1].file_list) == 3
    # ministacks[2] would have 5 (2 compressed + 3 real) but compressed are removed
    assert len(ministacks[2].file_list) == 3
    # ministacks[3] would have 6 (3 compressed + 3 real)
    # Only last 5 dates checked: 2 compressed removed, 1 compressed + 3 real remain
    assert len(ministacks[3].file_list) == 4
    # Later ministacks follow similar pattern - some compressed may remain
    # if not in the last 5 dates
    assert all(len(m.file_list) >= 3 for m in ministacks[3:-1])


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
    # With compressed SLCs removed from last 5 dates:
    # - Ministacks 0-2: all compressed SLCs removed → reference is first real SLC
    # - Ministack 3+: May have 1 compressed SLC remaining → reference is
    # that compressed SLC
    expected_ref_dates = []
    for _idx, ms in enumerate(ministacks):
        # If ministack has compressed SLCs, reference is the first (oldest) one
        # Otherwise, reference is the first real SLC
        if any(ms.is_compressed):
            # Has compressed SLC - use its date
            # The compressed_reference_idx points to a compressed SLC
            ref_date = ms.dates[ms.compressed_reference_idx][0]
        else:
            # No compressed SLCs - use first real SLC
            ref_date = ms.dates[0][0]
        expected_ref_dates.append(ref_date)

    for idx, c in enumerate(compslcs):
        assert c.reference_date == expected_ref_dates[idx]


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
