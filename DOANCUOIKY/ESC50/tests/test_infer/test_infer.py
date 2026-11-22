"""
tests/test_infer/test_infer.py - Unit test cho src/infer.py (Task H-03) và xuất lịch sử dự đoán ra CSV.
"""

import csv
from datetime import datetime
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import time
import wave

HERE = Path(__file__).resolve().parent
ROOT_DIR = next((p for p in [HERE, *HERE.parents] if p.name in ("ESC50", "DOANCUOIKY")), HERE.parent)
HISTORY_CSV = ROOT_DIR / "tests" / "test_infer" / "history.csv"

# Tự động tìm file infer.py ở bất kỳ vị trí nào trong dự án
infer_candidates = list(ROOT_DIR.rglob("infer.py"))
if not infer_candidates:
    raise FileNotFoundError(
        f"Không tìm thấy file 'infer.py' trong {ROOT_DIR}. "
        "Hãy kiểm tra lại bạn đã lưu file infer.py chưa!"
    )

INFER_PATH = infer_candidates[0]
if str(INFER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(INFER_PATH.parent))
if str(INFER_PATH.parent.parent) not in sys.path:
    sys.path.insert(0, str(INFER_PATH.parent.parent))

spec = importlib.util.spec_from_file_location("infer_module", str(INFER_PATH))
infer_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer_mod)

create_dummy_checkpoint = infer_mod.create_dummy_checkpoint
predict = infer_mod.predict


def _create_sample_wav(path: Path, sr: int, duration_s: float, channels: int = 1, freq: float = 440.0):
    """Tạo file WAV chuẩn 16-bit PCM bằng thư viện wave có sẵn của Python."""
    n_samples = int(sr * duration_s)
    frames = bytearray()
    for i in range(n_samples):
        val = 0.5 * math.sin(2.0 * math.pi * freq * (i / sr))
        sample_l = int(max(-32767, min(32767, val * 32767)))
        if channels == 2:
            sample_r = int(max(-32767, min(32767, val * 0.85 * 32767)))
            frames.extend(struct.pack("<hh", sample_l, sample_r))
        else:
            frames.extend(struct.pack("<h", sample_l))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))


def _append_history_csv(rows: list[dict], csv_path: Path = HISTORY_CSV) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = [
        "Ngày",
        "Tháng",
        "Giờ",
        "Tệp âm thanh",
        "Mô hình",
        "Số Mel bins",
        "Nhãn dự đoán",
        "Độ tin cậy (%)",
        "Thời gian xử lý (ms)",
    ]
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_infer_and_export_history():
    cases = [
        ("sample_dog_5s.wav", 20050, 5.0, 1, 340.0, "crnn", 128),      
        ("sample_rain_3s.wav", 12050, 3.2, 1, 880.0, "crnn", 64),      
        ("sample_siren_stereo.wav", 44100, 5.5, 2, 660.0, "crnn", 128),
        ("sample_clock_16k.wav", 16000, 4.0, 1, 523.25, "cnn", 128)
    ]

    history_rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for idx, (fname, sr, dur, ch, freq, arch, n_mels) in enumerate(cases, start=10):
            wav_path = tmp / fname
            ckpt_path = tmp / f"{arch}_{n_mels}.pt"
            _create_sample_wav(wav_path, sr=sr, duration_s=dur, channels=ch, freq=freq)

            dynamic_seed = int(time.time() * 1000) % 100000 + idx
            create_dummy_checkpoint(ckpt_path, arch=arch, n_mels=n_mels, seed=dynamic_seed)
            create_dummy_checkpoint(ckpt_path, arch=arch, n_mels=n_mels, seed=idx)

            t0 = time.perf_counter()
            out = predict(wav_path, ckpt_path, device="cpu", top_k=5, return_logmel=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert len(out["top5"]) == 5
            assert out["top1"] == out["top5"][0]
            assert out["logmel"].shape == (n_mels, 216)

            now = datetime.now()
            label, prob = out["top1"]
            history_rows.append({
                "Ngày": now.strftime("%d/%m/%Y"),
                "Tháng": now.strftime("%m/%Y"),
                "Giờ": now.strftime("%H:%M:%S"),
                "Tệp âm thanh": fname,
                "Mô hình": f"{out['arch'].upper()}-{out['n_mels']}",
                "Số Mel bins": out["n_mels"],
                "Nhãn dự đoán": label,
                "Độ tin cậy (%)": round(prob * 100, 2),
                "Thời gian xử lý (ms)": round(latency_ms, 2),
            })

    out_csv = _append_history_csv(history_rows, HISTORY_CSV)
    assert out_csv.exists()

    print("\n" + "=" * 96)
    print(f"{'Ngày':<12} {'Tháng':<9} {'Giờ':<10} {'Tệp âm thanh':<24} {'Mô hình':<10} {'Nhãn Top-1':<16} {'Tin cậy':<9} {'Latency'}")
    print("-" * 96)
    for r in history_rows:
        print(
            f"{r['Ngày']:<12} {r['Tháng']:<9} {r['Giờ']:<10} "
            f"{r['Tệp âm thanh']:<24} {r['Mô hình']:<10} {r['Nhãn dự đoán']:<16} "
            f"{r['Độ tin cậy (%)']:>6.2f}%   {r['Thời gian xử lý (ms)']:>6.1f} ms"
        )
    print("=" * 96)
    print(f"Đã xuất lịch sử CSV tại: {out_csv}")
    print("ALL H-03 TESTS OK")


if __name__ == "__main__":
    test_infer_and_export_history()