"""
src/infer.py - Module suy luận (inference) cho dự án phân loại âm thanh ESC-50 (Task H-03).
"""

import ast
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:
    from .models import build_model
except ImportError:
    from models import build_model

SRC_DIR = Path(__file__).resolve().parent


def _ensure_torchaudio_load_compat():
    """Đảm bảo torchaudio.load() chạy ổn định trên Windows kể cả khi thiếu torchcodec."""
    orig_load = torchaudio.load

    def _safe_load(uri, *args, **kwargs):
        try:
            return orig_load(uri, *args, **kwargs)
        except Exception as err:
            if "torchcodec" not in str(err).lower() and "No module named" not in str(err):
                raise
            try:
                import soundfile as sf

                data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T.copy()), int(sr)
            except ImportError:
                with wave.open(str(uri), "rb") as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw = wf.readframes(wf.getnframes())
                if sampwidth == 2:
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                arr = arr.reshape(-1, n_channels).T.copy()
                return torch.from_numpy(arr), int(sr)

    torchaudio.load = _safe_load


def _load_wav_to_logmel():
    """Nạp hàm wav_to_logmel từ src/data_lib.py hoặc src/data_lib (2).py (bỏ qua code chạy thử /content)."""
    _ensure_torchaudio_load_compat()

    if "gdown" not in sys.modules:
        try:
            import gdown  # noqa: F401
        except ImportError:
            sys.modules["gdown"] = types.ModuleType("gdown")

    try:
        from .data_lib import wav_to_logmel as fn
        return fn
    except ImportError:
        try:
            from data_lib import wav_to_logmel as fn
            return fn
        except ImportError:
            alt_path = SRC_DIR / "data_lib (2).py"
            if not alt_path.exists():
                raise ModuleNotFoundError("Không tìm thấy 'data_lib.py' hoặc 'data_lib (2).py' trong src/")

            # Chỉ biên dịch các câu lệnh import/hàm/lớp ở mức top-level, bỏ qua đoạn tải file /content ở cuối
            source = alt_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(alt_path))
            clean_body = [
                node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign))
            ]
            clean_mod = ast.Module(body=clean_body, type_ignores=[])
            ast.fix_missing_locations(clean_mod)

            namespace: Dict[str, Any] = {"__name__": "q03_data_lib", "__file__": str(alt_path)}
            exec(compile(clean_mod, str(alt_path), "exec"), namespace)
            return namespace["wav_to_logmel"]


wav_to_logmel = _load_wav_to_logmel()

ESC50_DEFAULT_CLASSES: List[str] = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
    "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
    "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn",
    "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw",
]


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], torch.device]:
    """Nạp checkpoint (.pt/.pth) theo schema Phần C9 và khởi tạo mô hình ở chế độ eval()."""
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_file}")

    target_device = _resolve_device(device)
    raw = torch.load(ckpt_file, map_location=target_device, weights_only=False)
    ckpt: Dict[str, Any] = raw if isinstance(raw, dict) and "state_dict" in raw else {"state_dict": raw}

    arch: str = str(ckpt.get("arch", "crnn")).lower()
    n_mels: int = int(ckpt.get("n_mels", 128))
    classes: List[str] = list(ckpt.get("classes", ESC50_DEFAULT_CLASSES))
    cfg: Dict[str, Any] = ckpt.get("cfg", {})
    dropout: float = float(cfg.get("dropout", 0.3))

    try:
        model = build_model(arch=arch, n_mels=n_mels, n_classes=len(classes), dropout=dropout)
    except TypeError:
        model = build_model(arch=arch, n_mels=n_mels)

    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()

    return model, ckpt, target_device


@torch.no_grad()
def predict(
    wav_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    top_k: int = 5,
    return_logmel: bool = False,
) -> Dict[str, Any]:
    """
    Suy luận nhãn âm thanh từ 1 file WAV bất kỳ.

    Trả về:
        {
            "top1": (label, prob),
            "top5": [(label_1, prob_1), ..., (label_5, prob_5)],
            "arch": str,
            "n_mels": int,
            "logmel": np.ndarray (nếu return_logmel=True)
        }
    """
    wav_file = Path(wav_path)
    if not wav_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file WAV: {wav_file}")

    model, ckpt, target_device = load_checkpoint(ckpt_path, device=device)
    arch: str = str(ckpt.get("arch", "crnn")).lower()
    n_mels: int = int(ckpt.get("n_mels", 128))
    mean: float = float(ckpt.get("mean", 0.0))
    std: float = float(ckpt.get("std", 1.0))
    classes: List[str] = list(ckpt.get("classes", ESC50_DEFAULT_CLASSES))

    # Trích xuất Log-Mel [1, n_mels, 216] (mono, resample 22050Hz, cắt/pad 5s)
    logmel = wav_to_logmel(wav_file, n_mels=n_mels, device=str(target_device))

    # Chuẩn hóa bằng mean/std lưu trong checkpoint (Phần C2)
    logmel_norm = (logmel - mean) / (std + 1e-8)

    # Thêm chiều batch -> [1, 1, n_mels, 216]
    x = logmel_norm.unsqueeze(0)

    # Suy luận & Softmax
    logits = model(x)
    probs = F.softmax(logits, dim=-1).squeeze(0)

    k = min(top_k, len(classes))
    top_probs, top_indices = torch.topk(probs, k=k)
    top_list: List[Tuple[str, float]] = [
        (classes[int(idx)], float(prob))
        for prob, idx in zip(top_probs.cpu(), top_indices.cpu())
    ]

    result: Dict[str, Any] = {
        "top1": top_list[0],
        "top5": top_list,
        "arch": arch,
        "n_mels": n_mels,
    }

    if return_logmel:
        result["logmel"] = logmel.squeeze(0).cpu().numpy()

    return result


def create_dummy_checkpoint(
    save_path: Union[str, Path],
    arch: str = "crnn",
    n_mels: int = 128,
    seed: int = 42,
) -> Path:
    """Tạo checkpoint giả (khởi tạo ngẫu nhiên) đúng chuẩn Phần C9 để test khi chưa có checkpoint thật."""
    torch.manual_seed(seed)
    model = build_model(arch=arch, n_mels=n_mels)
    model.eval()

    ckpt = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "n_mels": n_mels,
        "mean": -4.5,
        "std": 3.2,
        "classes": ESC50_DEFAULT_CLASSES,
        "cfg": {
            "seed": seed,
            "dropout": 0.3,
        },
    }

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return out_path