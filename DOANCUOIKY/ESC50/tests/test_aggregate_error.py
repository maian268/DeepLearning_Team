"""Test cho src/aggregate.py (A-04) và src/error_analysis.py (A-05) bằng JSON giả.
Chạy: pytest tests/test_aggregate_error.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src import eval_lib as ev
from src import aggregate as ag
from src import error_analysis as ea


def test_aggregate_creates_tables_and_figure(tmp_path):
    res = str(tmp_path / "res")
    ev.make_fake_results(res, exps=(1, 2, 3, 4, 5))
    out = ag.aggregate((1, 2, 3, 4, 5), res, str(tmp_path / "tab"), str(tmp_path / "fig"), fake=True)

    for p in out["files"].values():
        assert os.path.exists(p)
    t = out["tables"]
    assert list(t["results"]["EXP"]) == ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5"]
    assert len(t["rq"]) == 5                      # 5 cặp so sánh cho RQ1-RQ3
    assert len(t["gap"]) == 5 and len(t["cost"]) == 5


def test_error_analysis_runs_without_cache(tmp_path):
    res = str(tmp_path / "res")
    ev.make_fake_results(res, exps=(1, 2, 3, 4, 5))
    out = ea.run(None, (1, 2, 3, 4, 5), res, str(tmp_path / "no_cache"), None,
                 str(tmp_path / "fig"), str(tmp_path / "tab"), fake=True)
    assert os.path.exists(out["confmat"]) and os.path.exists(out["f1_per_class"])
    assert len(out["pairs"]) == 10
    assert out["pair_figs"] == []                 # chưa có cache -> bỏ qua hình spectrogram


def test_error_analysis_pair_figures_with_logmel(tmp_path):
    res = str(tmp_path / "res")
    ev.make_fake_results(res, exps=(5,))
    fake_logmel = lambda fname: np.random.default_rng(0).normal(size=(128, 216))
    out = ea.run(5, (5,), res, str(tmp_path / "c"), None, str(tmp_path / "fig"),
                 str(tmp_path / "tab"), fake=True, get_logmel=fake_logmel)
    assert len(out["pair_figs"]) == 5
    assert all(os.path.exists(p) for p in out["pair_figs"])
