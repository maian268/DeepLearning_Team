"""tests/test_run_experiment.py - test run_experiment (T-03, Tấn) bằng cache giả.

Cache giả nhỏ (5 fold x 50 mẫu, 50 lớp) cùng định dạng C1, chạy được trên CPU:
    1. chạy 1 fold: JSON và checkpoint có đủ trường theo C9
    2. chia fold đúng C3 (val = k mod 5 + 1, tách rời, đủ mẫu)
    3. bỏ qua fold đã có JSON (không train lại)
    4. specaug=True mà thiếu specaug_fn thì báo lỗi rõ ràng
    5. split_fn (kiểu get_splits) cho cùng dữ liệu test như split_indices
    6. load_config gộp YAML đúng thứ tự

Chạy:
    python test_run_experiment.py
    pytest test_run_experiment.py -v
Cần scikit-learn (pip install scikit-learn); test 5 cần pyyaml (pip install pyyaml).
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_lib
from train_lib import CFG, load_config, run_experiment, split_indices

N_MELS = 64
PER_FOLD = 50
N = 5 * PER_FOLD


def _make_fake_cache(cache_dir):
    """Cache giả theo C1: X, y, fold, filenames, classes (cùng shape/dtype)."""
    os.makedirs(cache_dir, exist_ok=True)
    torch.manual_seed(0)
    y = torch.arange(N) % 50
    cache = dict(
        X=torch.randn(N, 1, N_MELS, 216, dtype=torch.float32),
        y=y.to(torch.int64),
        fold=(torch.arange(N) // PER_FOLD + 1).to(torch.int64),   # mỗi fold 50 mẫu
        filenames=[f"fake_{i}.wav" for i in range(N)],
        classes=[f"class_{i}" for i in range(50)],
    )
    torch.save(cache, os.path.join(cache_dir, f"logmel{N_MELS}.pt"))
    return cache


def test_split_follows_c3():
    fold = torch.arange(N) // PER_FOLD + 1
    for k in range(1, 6):
        tr, va, te = split_indices(fold, k)
        val_fold = k % 5 + 1
        assert set(fold[te].tolist()) == {k}
        assert set(fold[va].tolist()) == {val_fold}
        assert k not in fold[tr].tolist() and val_fold not in fold[tr].tolist()
        assert len(tr) == 3 * PER_FOLD and len(va) == len(te) == PER_FOLD
        assert len(set(tr.tolist()) & set(va.tolist()) & set(te.tolist())) == 0


def test_run_experiment_output_and_resume():
    with tempfile.TemporaryDirectory() as d:
        cache_dir = os.path.join(d, "cache")
        res_dir = os.path.join(d, "results")
        ck_dir = os.path.join(d, "checkpoints")
        cache = _make_fake_cache(cache_dir)
        cfg = dict(CFG, max_epochs=2)
        kw = dict(cache_dir=cache_dir, results_dir=res_dir, ckpt_dir=ck_dir,
                  cfg=cfg, verbose=False)

        r = run_experiment(2, "CNN", N_MELS, False, 3, **kw)

        # JSON đủ trường theo C9
        json_path = os.path.join(res_dir, "exp2_fold3.json")
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        expected = {"exp_id", "arch", "n_mels", "specaug", "test_fold", "val_fold",
                    "seed", "params", "best_epoch", "train_time_s", "val_acc_best",
                    "acc", "macro_f1", "filenames", "labels", "preds", "probs",
                    "history"}
        assert set(data) == expected, set(data) ^ expected
        assert data["val_fold"] == 4 and data["test_fold"] == 3
        assert data["params"] == 1_185_490
        assert len(data["labels"]) == len(data["preds"]) == PER_FOLD
        assert len(data["probs"]) == PER_FOLD and len(data["probs"][0]) == 50
        assert len(data["filenames"]) == PER_FOLD
        assert 1 <= len(data["history"]) <= 2
        assert 0.0 <= data["acc"] <= 1.0 and 0.0 <= data["macro_f1"] <= 1.0
        assert abs(sum(data["probs"][0]) - 1.0) < 1e-3     # softmax
        # nhãn trong JSON là nhãn của fold test (fold 3)
        assert data["labels"] == cache["y"][100:150].tolist()

        # Checkpoint đủ trường theo C9
        ck = torch.load(os.path.join(ck_dir, "exp2_fold3.pt"), weights_only=False)
        assert set(ck) == {"state_dict", "arch", "n_mels", "mean", "std",
                           "classes", "cfg"}
        assert ck["arch"] == "CNN" and ck["n_mels"] == N_MELS
        assert len(ck["classes"]) == 50

        # Chạy lại: phải bỏ qua, không gọi fit()
        original_fit = train_lib.fit

        def must_not_run(*a, **k):
            raise AssertionError("fit() bị gọi lại dù đã có JSON")

        train_lib.fit = must_not_run
        try:
            r2 = run_experiment(2, "CNN", N_MELS, False, 3, **kw)
        finally:
            train_lib.fit = original_fit
        assert r2["acc"] == r["acc"] and r2["params"] == r["params"]


def _stub_get_splits(calls):
    def fn(cache, test_fold):
        calls.append(test_fold)
        fold = cache["fold"]
        val_fold = test_fold % 5 + 1

        def part(mask):
            return {"X": cache["X"][mask], "y": cache["y"][mask],
                    "indices": torch.where(mask)[0]}

        train_mask = (fold != test_fold) & (fold != val_fold)
        return part(train_mask), part(fold == val_fold), part(fold == test_fold)
    return fn


def test_run_experiment_with_split_fn():
    with tempfile.TemporaryDirectory() as d:
        cache_dir = os.path.join(d, "cache")
        _make_fake_cache(cache_dir)
        cfg = dict(CFG, max_epochs=1)
        calls = []
        outs = {}
        for tag, split_fn in (("a", None), ("b", _stub_get_splits(calls))):
            outs[tag] = run_experiment(
                2, "CNN", N_MELS, False, 3, cfg=cfg, cache_dir=cache_dir,
                results_dir=os.path.join(d, tag, "results"),
                ckpt_dir=os.path.join(d, tag, "checkpoints"),
                split_fn=split_fn, verbose=False)
        assert calls == [3], calls
        assert outs["a"]["filenames"] == outs["b"]["filenames"]
        assert outs["a"]["labels"] == outs["b"]["labels"]
        assert outs["b"]["val_fold"] == 4


def test_specaug_requires_fn():
    with tempfile.TemporaryDirectory() as d:
        _make_fake_cache(os.path.join(d, "cache"))
        try:
            run_experiment(4, "CNN", N_MELS, True, 1,
                           cache_dir=os.path.join(d, "cache"),
                           results_dir=os.path.join(d, "r"),
                           ckpt_dir=os.path.join(d, "c"),
                           cfg=dict(CFG, max_epochs=1), verbose=False)
        except ValueError:
            return
        raise AssertionError("đáng lẽ phải báo ValueError khi thiếu specaug_fn")


def test_load_config_merges_yaml():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.yaml")
        exp = os.path.join(d, "exp2.yaml")
        with open(base, "w", encoding="utf-8") as f:
            f.write("batch_size: 16\nseed: 7\nlr: 1e-3\n")   # PyYAML đọc 1e-3 thành chuỗi
        with open(exp, "w", encoding="utf-8") as f:
            f.write("arch: CNN\nn_mels: 64\nspecaug: true\nseed: 8\n")
        cfg = load_config(base, exp)
        assert cfg["batch_size"] == 16 and cfg["seed"] == 8   # file sau ghi đè
        assert isinstance(cfg["lr"], float) and cfg["lr"] == 1e-3   # chuỗi "1e-3" đã ép thành số
        assert cfg["weight_decay"] == CFG["weight_decay"]      # khóa không có giữ mặc định
        assert cfg["arch"] == "CNN" and cfg["n_mels"] == 64 and cfg["specaug"] is True


if __name__ == "__main__":
    tests = [test_split_follows_c3, test_run_experiment_output_and_resume,
             test_run_experiment_with_split_fn,
             test_specaug_requires_fn, test_load_config_merges_yaml]
    for i, t in enumerate(tests, 1):
        t()
        print(f"{i}/{len(tests)} {t.__name__} OK")
    print("ALL OK")
