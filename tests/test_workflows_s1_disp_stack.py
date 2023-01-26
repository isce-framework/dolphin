import random
from itertools import chain
from pathlib import Path

from dolphin.workflows import s1_disp_stack


def test_group_by_burst():
    expected = {
        "t087_185678_iw2": [
            Path("t087_185678_iw2/20180210/t087_185678_iw2_20180210_VV.h5"),
            Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318_VV.h5"),
            Path("t087_185678_iw2/20180423/t087_185678_iw2_20180423_VV.h5"),
        ],
        "t087_185678_iw3": [
            Path("t087_185678_iw3/20180210/t087_185678_iw3_20180210_VV.h5"),
            Path("t087_185678_iw3/20180318/t087_185678_iw3_20180318_VV.h5"),
            Path("t087_185678_iw3/20180517/t087_185678_iw3_20180517_VV.h5"),
        ],
        "t087_185679_iw1": [
            Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            Path("t087_185679_iw1/20180318/t087_185679_iw1_20180318_VV.h5"),
        ],
    }
    in_files = list(chain.from_iterable(expected.values()))

    assert s1_disp_stack._group_by_burst(in_files) == expected

    # Any order should work
    random.shuffle(in_files)
    # but the order of the lists of each key may be different
    for burst, file_list in s1_disp_stack._group_by_burst(in_files).items():
        assert sorted(file_list) == sorted(expected[burst])
