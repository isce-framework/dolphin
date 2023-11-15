from datetime import datetime
from pathlib import Path

import pytest

from dolphin.stack import MiniStackInfo, MiniStackPlanner

NUM_ACQ = 10


# Note: uses the fixtures from conftest.py
# Use a smaller subset for these tests
@pytest.fixture
def files(slc_file_list):
    return slc_file_list[:NUM_ACQ]


@pytest.fixture
def dates(slc_date_list):
    return slc_date_list[:NUM_ACQ]


@pytest.fixture
def is_compressed(files):
    return [False] * len(files[:NUM_ACQ])


@pytest.fixture
def date_tuples(dates):
    # To mimic what we get back from `get_dates`
    return [(d,) for d in dates]


def test_create_ministack(tmp_path, files, date_tuples, is_compressed):
    MiniStackInfo(files, is_compressed=is_compressed, dates=date_tuples)

    # Check it works by providing a real local dir
    MiniStackInfo(
        files,
        dates=date_tuples,
        is_compressed=is_compressed,
        output_folder=tmp_path,
    )

    # Check it works when making up a future output dir which doesn't exist
    MiniStackInfo(
        files,
        dates=date_tuples,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
    )


def test_mismatched_lengths(files, date_tuples, is_compressed):
    with pytest.raises(ValueError):
        MiniStackInfo(files, dates=date_tuples[1:], is_compressed=is_compressed)

    with pytest.raises(ValueError):
        MiniStackInfo(files, dates=date_tuples, is_compressed=is_compressed[:-1])


def test_create_ministack_dates(files, dates, is_compressed):
    # Check we can provide a list of dates instead of tuples
    m = MiniStackInfo(files, dates=dates, is_compressed=is_compressed)
    assert m.dates[0] == (dates[0],)


@pytest.fixture
def ministack(files, date_tuples, is_compressed):
    return MiniStackInfo(
        files,
        dates=date_tuples,
        is_compressed=is_compressed,
        output_folder="fake_dir",
    )


def test_ministack_attrs(ministack, dates):
    assert ministack.full_date_range == (dates[0], dates[-1])
    assert ministack.real_slc_date_range_str == "20220101_20220110"


def test_create_compressed_slc(files, date_tuples, is_compressed):
    m = MiniStackInfo(files, is_compressed=is_compressed, dates=date_tuples)
    comp_slc = m.get_compressed_slc_info()
    assert comp_slc.real_slc_dates == date_tuples
    assert comp_slc.real_slc_file_list == files

    # Now try one where we marked the first of the stack as compressed
    is_compressed2 = is_compressed.copy()
    is_compressed2[0] = True
    m2 = MiniStackInfo(files, is_compressed=is_compressed2, dates=date_tuples)
    comp_slc2 = m2.get_compressed_slc_info()
    assert comp_slc2.real_slc_dates == date_tuples[1:]
    assert comp_slc2.real_slc_file_list == files[1:]
    assert comp_slc2.compressed_slc_file_list == files[0:1]


def test_ministack_planner(files, date_tuples, is_compressed):
    msp = MiniStackPlanner(
        files,
        dates=date_tuples,
        is_compressed=is_compressed,
        output_folder=Path("fake_dir"),
        max_num_compressed=5,
    )
    ms_list = msp.plan(4)
    assert len(ms_list) == 3

    with pytest.warns(UserWarning):
        ms_list = msp.plan(3)

    assert len(ms_list) == 4
    assert ms_list[0].file_list == files[:3]
    assert ms_list[1].file_list == files[3:6]
    assert ms_list[2].file_list == files[6:9]
    assert ms_list[3].file_list == files[9:10]

    assert ms_list[0].output_folder == Path("fake_dir/20220101_20220103")
    assert ms_list[1].output_folder == Path("fake_dir/20220104_20220106")
    assert ms_list[2].output_folder == Path("fake_dir/20220107_20220109")
    assert ms_list[3].output_folder == Path("fake_dir/20220110_20220110")

    assert all([ms.reference_date == datetime(2022, 1, 1) for ms in ms_list])


# """Result above is:

# [
#     MiniStackInfo(
#         file_list=[
#             PosixPath('pathtest/gtiff/20220101.slc.tif'),
#             PosixPath('pathtest/gtiff/20220102.slc.tif'),
#             PosixPath('pathtest/gtiff/20220103.slc.tif')
#         ],
#         dates=[(datetime.datetime(2022, 1, 1, 0, 0),), (datetime.datetime(2022, 1, 2, 0, 0),), (datetime.datetime(2022, 1, 3, 0, 0),)],
#         is_compressed=[False, False, False],
#         reference_date=datetime.datetime(2022, 1, 1, 0, 0),
#         file_date_fmt='%Y%m%d',
#         output_folder=PosixPath('20220101_20220103'),
#         reference_idx=0
#     ),
#     MiniStackInfo(
#         file_list=[
#             PosixPath('pathtest/gtiff/20220104.slc.tif'),
#             PosixPath('pathtest/gtiff/20220105.slc.tif'),
#             PosixPath('pathtest/gtiff/20220106.slc.tif')
#         ],
#         dates=[(datetime.datetime(2022, 1, 4, 0, 0),), (datetime.datetime(2022, 1, 5, 0, 0),), (datetime.datetime(2022, 1, 6, 0, 0),)],
#         is_compressed=[False, False, False],
#         reference_date=datetime.datetime(2022, 1, 1, 0, 0),
#         file_date_fmt='%Y%m%d',
#         output_folder=PosixPath('20220104_20220106'),
#         reference_idx=0
#     ),
#     MiniStackInfo(
#         file_list=[
#             PosixPath('pathtest/gtiff/20220107.slc.tif'),
#             PosixPath('pathtest/gtiff/20220108.slc.tif'),
#             PosixPath('pathtest/gtiff/20220109.slc.tif')
#         ],
#         dates=[(datetime.datetime(2022, 1, 7, 0, 0),), (datetime.datetime(2022, 1, 8, 0, 0),), (datetime.datetime(2022, 1, 9, 0, 0),)],
#         is_compressed=[False, False, False],
#         reference_date=datetime.datetime(2022, 1, 1, 0, 0),
#         file_date_fmt='%Y%m%d',
#         output_folder=PosixPath('20220107_20220109'),
#         reference_idx=0
#     ),
#     MiniStackInfo(
#         file_list=[PosixPath('pathtest/gtiff/20220110.slc.tif')],
#         dates=[(datetime.datetime(2022, 1, 10, 0, 0),)],
#         is_compressed=[False],
#         reference_date=datetime.datetime(2022, 1, 1, 0, 0),
#         file_date_fmt='%Y%m%d',
#         output_folder=PosixPath('20220110_20220110'),
#         reference_idx=0
#     )
# ]
# """
