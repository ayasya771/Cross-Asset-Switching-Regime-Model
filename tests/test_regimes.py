"""Regime post-processing: label smoothing and the observable-driven naming."""
import numpy as np

from regimelab.regimes.extract import _regime_names, modal_smooth


def test_modal_smooth_removes_single_day_flicker():
    labels = np.array([0, 0, 0, 1, 0, 0, 0])
    out = modal_smooth(labels, window=5)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0]


def test_modal_smooth_keeps_genuine_runs():
    labels = np.array([0] * 10 + [1] * 10)
    out = modal_smooth(labels, window=5)
    assert out[:8].tolist() == [0] * 8
    assert out[12:].tolist() == [1] * 8
    assert set(out.tolist()) == {0, 1}


def test_modal_smooth_window_one_is_identity():
    labels = np.array([0, 1, 0, 2, 2, 1])
    assert modal_smooth(labels, window=1).tolist() == labels.tolist()


def test_modal_smooth_preserves_length_and_label_set():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    out = modal_smooth(labels, window=7)
    assert len(out) == len(labels)
    assert set(out.tolist()) <= set(labels.tolist())


def _row(**kw):
    base = dict(vix=15.0, hy_oas=4.0, slope=1.5, ann_ret=8.0)
    base.update(kw)
    return base


def test_regime_names_anchor_calmest_and_most_volatile():
    rows = [_row(vix=12), _row(vix=20), _row(vix=40)]
    names = _regime_names(rows)
    assert names[0] == "Calm Risk-On"
    assert names[-1] == "Crisis / Stress"
    assert len(names) == 3


def test_regime_names_read_the_middle_states_from_observables():
    flat_curve = _row(slope=0.1)
    drawdown = _row(slope=2.0, ann_ret=-15.0)
    rows = [_row(vix=12), flat_curve, drawdown, _row(vix=40)]
    names = _regime_names(rows)
    assert names[1] == "Late-Cycle / Flat Curve"
    assert names[2] == "Risk-Off / Drawdown"


def test_regime_names_are_unique():
    rows = [_row(vix=12), _row(slope=0.1), _row(slope=0.1), _row(vix=40)]
    names = _regime_names(rows)
    assert len(set(names)) == len(names)
