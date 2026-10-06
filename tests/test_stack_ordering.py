"""A compressed SLC referenced to a date among the reals is stacked once."""

from datetime import datetime, timedelta

import pytest

from dolphin.stack import CompressedSlcPlan, MiniStackPlanner

D0 = datetime(2019, 8, 13)
STEP = timedelta(days=6)


def _real(i: int) -> str:
    return f"t042_088921_iw1_{(D0 + i * STEP):%Y%m%d}.h5"


def _comp(ref_i: int, start_i: int = 0) -> str:
    a, b = D0 + start_i * STEP, D0 + ref_i * STEP
    return f"compressed_t042_088921_iw1_{b:%Y%m%d}_{a:%Y%m%d}_{b:%Y%m%d}.h5"


def _plan(files, size=100, **kw):
    is_comp = ["compressed" in f for f in files]
    dates = []
    for f, _c in zip(files, is_comp, strict=False):
        parts = [
            p for p in f.replace(".h5", "").split("_") if p.isdigit() and len(p) == 8
        ]
        dates.append([datetime.strptime(p, "%Y%m%d") for p in parts])
    planner = MiniStackPlanner(
        file_list=files,
        dates=dates,
        is_compressed=is_comp,
        output_folder="out",
        max_num_compressed=kw.pop("max_num_compressed", 100),
        compressed_slc_plan=kw.pop("plan", CompressedSlcPlan.LAST_PER_MINISTACK),
        **kw,
    )
    return planner.plan(size)


def test_historical_shape_is_unchanged():
    """Compressed SLCs before every real date."""
    files = [_comp(-30), _comp(-20), *(_real(i) for i in range(15))]
    (ms,) = _plan(files)
    assert [str(f) for f in ms.file_list] == files
    assert list(ms.is_compressed) == [True, True] + [False] * 15
    assert ms.first_real_slc_idx == 2


def test_an_interleaved_compressed_slc_is_not_duplicated():
    """Newest compressed SLC referenced to a date among the reals."""
    files = [
        _comp(-30),
        *(_real(i) for i in range(10)),
        _comp(10),
        *(_real(i) for i in range(11, 15)),
    ]
    files.sort(key=lambda f: f.split("_")[-1])  # as dolphin sorts them
    (ms,) = _plan(files)
    names = [str(f) for f in ms.file_list]
    assert len(names) == len(set(names)), "no input may enter the stack twice"
    assert len(names) == len(files)
    assert list(ms.is_compressed) == [True, True] + [False] * 14
    assert all("compressed" in n for n in names[:2])
    assert not any("compressed" in n for n in names[2:])


def test_the_reference_index_lands_on_the_newest_compressed():
    files = [
        _comp(-30),
        *(_real(i) for i in range(10)),
        _comp(10),
        *(_real(i) for i in range(11, 15)),
    ]
    files.sort(key=lambda f: f.split("_")[-1])
    (ms,) = _plan(files, output_reference_idx=1)
    ref = str(ms.file_list[ms.output_reference_idx])
    assert ms.is_compressed[ms.output_reference_idx]
    assert ref == _comp(10), "the newest compressed SLC, not a real date"


@pytest.mark.parametrize("size", [5, 15])
def test_chunking_counts_real_slcs(size):
    """`ministack_size` counts real SLCs. With compressed files at the front
    that is what slicing from `first_real_slc_idx` already did; it must not
    change."""
    files = [_comp(-30), *(_real(i) for i in range(30))]
    # `last_per_ministack` refuses to plan several ministacks at once.
    stacks = _plan(files, size, plan=CompressedSlcPlan.ALWAYS_FIRST)
    assert len(stacks) == 30 // size
    for ms in stacks:
        assert sum(not c for c in ms.is_compressed) == size


def test_a_stack_with_no_real_slcs_is_rejected():
    with pytest.raises(ValueError, match="No real SLCs"):
        _plan([_comp(-30), _comp(-20)])
