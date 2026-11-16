"""Manual-index networks and the compressed SLC's reference epoch."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from dolphin.workflows.config import InterferogramNetwork
from dolphin.workflows.wrapped_phase import compressed_reference_ifgs, create_ifgs

NEAREST_4 = [
    (-2, -1),
    (-3, -1),
    (-4, -1),
    (-3, -2),
    (-4, -2),
    (-4, -3),
    (-5, -1),
    (-5, -2),
    (-5, -3),
    (-5, -4),
]
D0 = datetime(2019, 8, 13)


def _dates(n, skip=()):
    out = [D0 + timedelta(days=6 * k) for k in range(n)]
    return [d for k, d in enumerate(out) if k not in skip]


def _ifgs(ref, dates):
    return [Path(f"{ref:%Y%m%d}_{d:%Y%m%d}.int.vrt") for d in dates]


def test_nothing_added_when_the_reference_predates_the_window():
    """Compressed epoch before the window."""
    dates = _dates(14)
    ref = D0 - timedelta(days=6)
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates)) == []


def test_the_missing_last_interval_is_added():
    """Reference is the second-to-last date."""
    all_dates = _dates(15)
    ref = all_dates[-2]
    dates = [d for d in all_dates if d != ref]
    got = compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates))
    assert got == [Path(f"{ref:%Y%m%d}_{all_dates[-1]:%Y%m%d}.int.vrt")]


def test_nothing_added_when_the_reference_is_deeper_in_the_window():
    """The newest interval is already a real-to-real edge."""
    all_dates = _dates(15)
    ref = all_dates[-4]
    dates = [d for d in all_dates if d != ref]
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates)) == []


def test_exactly_one_pair_is_ever_added():
    all_dates = _dates(15)
    ref = all_dates[-2]
    dates = [d for d in all_dates if d != ref]
    for idx in (NEAREST_4, [(-2, -1)], [(-9, -1), (-2, -1)]):
        got = compressed_reference_ifgs(idx, ref, dates, _ifgs(ref, dates))
        assert got == [Path(f"{ref:%Y%m%d}_{all_dates[-1]:%Y%m%d}.int.vrt")], idx


def test_a_reference_after_every_real_date_adds_nothing():
    """A reference that is the newest date yields nothing."""
    dates = _dates(14)
    ref = dates[-1] + timedelta(days=6)
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates)) == []


def test_positive_indexes_are_not_guessed_at():
    dates = _dates(14)
    ref = dates[-2]
    assert compressed_reference_ifgs([(0, 1)], ref, dates, _ifgs(ref, dates)) == []


def test_datetime_and_date_compare_as_dates():
    """reference_date carries a time of day; filename dates do not."""
    all_dates = _dates(15)
    ref = all_dates[-2] + timedelta(hours=14, minutes=8)
    dates = [d for d in all_dates if d != all_dates[-2]]
    assert len(compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates))) == 1


def _phase_linked(tmp_path, dates):
    out = []
    for d in dates:
        p = tmp_path / f"{d:%Y%m%d}.slc.tif"
        p.touch()
        out.append(p)
    return out


@pytest.mark.parametrize("compressed", [False, True])
def test_create_ifgs_adds_the_reference_ifg_only_with_compressed_slcs(
    tmp_path, compressed
):
    all_dates = _dates(15)
    ref = all_dates[-2]
    pl = _phase_linked(tmp_path, [d for d in all_dates if d != ref])
    net = InterferogramNetwork(indexes=NEAREST_4)
    net._directory = tmp_path / "ifgs"
    names = {p.name for p in create_ifgs(net, pl, compressed, ref, dry_run=True)}
    wanted = f"{ref:%Y%m%d}_{all_dates[-1]:%Y%m%d}"
    assert any(n.startswith(wanted) for n in names) is compressed
    assert len(names) == 10 + compressed


# --- minimal anchor: one edge, always -------------------------------------


@pytest.mark.parametrize("position", [1, 2, 3, 5, 9])
def test_anchor_keeps_exactly_one_edge_at_every_position(position):
    all_dates = _dates(20)
    ref = all_dates[-(position + 1)]
    dates = [d for d in all_dates if d != ref]
    got = compressed_reference_ifgs(
        NEAREST_4, ref, dates, _ifgs(ref, dates), anchor=True
    )
    assert len(got) == 1, (position, got)


def test_anchor_picks_the_shortest_baseline_pair():
    all_dates = _dates(20)
    ref = all_dates[-4]  # position 3: inside NEAREST_4's window
    dates = [d for d in all_dates if d != ref]
    got = compressed_reference_ifgs(
        NEAREST_4, ref, dates, _ifgs(ref, dates), anchor=True
    )
    assert got == _ifgs(ref, [all_dates[-3]])


def test_anchor_matches_the_default_pair_at_the_second_to_last_position():
    all_dates = _dates(20)
    ref = all_dates[-2]
    dates = [d for d in all_dates if d != ref]
    ifgs = _ifgs(ref, dates)
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, ifgs, anchor=True) == (
        compressed_reference_ifgs(NEAREST_4, ref, dates, ifgs)
    )


def test_anchor_off_adds_nothing_away_from_the_second_to_last_position():
    all_dates = _dates(20)
    ref = all_dates[-4]
    dates = [d for d in all_dates if d != ref]
    assert compressed_reference_ifgs(NEAREST_4, ref, dates, _ifgs(ref, dates)) == []


def test_anchor_is_an_interferogram_network_option():
    net = InterferogramNetwork(indexes=NEAREST_4, compressed_reference_anchor=True)
    assert net.compressed_reference_anchor is True
    assert InterferogramNetwork(indexes=NEAREST_4).compressed_reference_anchor is False
