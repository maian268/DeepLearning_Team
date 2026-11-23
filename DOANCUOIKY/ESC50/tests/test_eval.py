"""Test cho src/eval_lib.py bằng JSON giả (A-03). Chạy: pytest tests/test_eval.py"""
import numpy as np
from src import eval_lib as ev


def test_pipeline_with_fake_results(tmp_path):
    out = str(tmp_path / "fake")
    ev.make_fake_results(out, exps=(1, 2, 3, 4, 5))

    rs = ev.load_exp(1, result_dir=out)
    assert sorted(rs) == [1, 2, 3, 4, 5]
    for r in rs.values():
        assert ev.check_consistency(r)           # acc, macro_f1 tính lại khớp JSON

    cm = ev.confusion_pooled(rs)
    assert cm.shape == (50, 50) and cm.sum() == 2000
    assert np.allclose(ev.confusion_pooled(rs, normalize=True).sum(axis=1), 1.0)

    df, per_fold = ev.summarize([1, 2, 3, 4, 5], out)
    assert list(df["EXP"]) == ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5"]
    assert (df["Số fold"] == 5).all()

    rq = ev.rq_table(per_fold)
    assert len(rq) == 5

    pairs = ev.top_confused_pairs(cm, ev.DEFAULT_CLASS_NAMES, k=10)
    assert len(pairs) == 10 and pairs["Số lần nhầm"].is_monotonic_decreasing


def test_missing_folds_are_reported(tmp_path):
    out = str(tmp_path / "fake")
    ev.make_fake_results(out, exps=(1,))
    avail = ev.availability_table([1, 2], out)
    assert avail.loc[0, "Số fold"] == 5 and avail.loc[1, "Số fold"] == 0
