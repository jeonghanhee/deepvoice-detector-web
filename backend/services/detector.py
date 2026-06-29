from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import whisper

from ..config import HEATMAP_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PathLike = Union[str, os.PathLike[str]]


def _resolve_project_path(path_value: PathLike) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DEEPVOICE_SOURCE_DIR = _resolve_project_path(
    os.getenv("DEEPVOICE_SOURCE_DIR", "external/deepvoice-detection")
)
DEFAULT_TTS_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model_tts_whisper_encoder_lcnn.pt"
DEFAULT_RVC_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model_rvc_whisper_encoder_lcnn.pt"
TYPO_RVC_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model_rvc_whipser_encoder_lcnn.pt"
LEGACY_TTS_MODEL_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_tts_whisper_encoder_lcnn.pt"
LEGACY_RVC_MODEL_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_rvc_whisper_encoder_lcnn.pt"
LEGACY_TYPO_RVC_MODEL_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model_rvc_whipser_encoder_lcnn.pt"

DEFAULT_MODEL_PATH = DEFAULT_TTS_MODEL_PATH
LEGACY_MODEL_PATH = LEGACY_TTS_MODEL_PATH
DEFAULT_THRESHOLD = 0.28

_MODEL_ENV_VARS = {
    "tts": ("DEEPVOICE_TTS_MODEL_PATH", "DEEPVOICE_MODEL_PATH"),
    "rvc": ("DEEPVOICE_RVC_MODEL_PATH",),
}

_MODEL_PATH_CANDIDATES = {
    "tts": (DEFAULT_TTS_MODEL_PATH, LEGACY_TTS_MODEL_PATH),
    "rvc": (
        DEFAULT_RVC_MODEL_PATH,
        TYPO_RVC_MODEL_PATH,
        LEGACY_RVC_MODEL_PATH,
        LEGACY_TYPO_RVC_MODEL_PATH,
    ),
}

_MODEL_LABELS = {
    "tts": "TTS",
    "rvc": "RVC",
}

_MODEL_CACHE: dict[tuple[str, Path, str], tuple[torch.nn.Module, float]] = {}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: Path


@lru_cache(maxsize=1)
def _load_deepvoice_model_class():
    model_file = DEEPVOICE_SOURCE_DIR / "model" / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(
            "DeepVoice submodule is missing. Run "
            "`git submodule update --init --recursive` from the project root."
        )

    spec = importlib.util.spec_from_file_location("deepvoice_detection_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import DeepVoice model module: {model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WhisperEncoderLCNN


def _resolve_model_path(
    model_path: Optional[PathLike] = None,
    model_key: str = "tts",
) -> Path:
    explicit_path = model_path
    if explicit_path:
        path = _resolve_project_path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return path

    if model_key not in _MODEL_PATH_CANDIDATES:
        raise ValueError(f"Unknown model key: {model_key}")

    for env_name in _MODEL_ENV_VARS[model_key]:
        env_value = os.getenv(env_name)
        if env_value:
            path = _resolve_project_path(env_value)
            if not path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint from {env_name} not found: {path}"
                )
            return path

    for path in _MODEL_PATH_CANDIDATES[model_key]:
        if path.exists():
            return path.resolve()

    candidates = ", ".join(str(path) for path in _MODEL_PATH_CANDIDATES[model_key])
    raise FileNotFoundError(
        f"{_MODEL_LABELS[model_key]} model checkpoint not found. Put the .pt file at "
        f"{candidates} or set one of: {', '.join(_MODEL_ENV_VARS[model_key])}."
    )


def _resolve_optional_model_path(model_key: str) -> Optional[Path]:
    try:
        return _resolve_model_path(model_key=model_key)
    except FileNotFoundError:
        if any(os.getenv(env_name) for env_name in _MODEL_ENV_VARS[model_key]):
            raise
        return None


def _available_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            key="tts",
            label=_MODEL_LABELS["tts"],
            path=_resolve_model_path(model_key="tts"),
        )
    ]

    rvc_path = _resolve_optional_model_path("rvc")
    if rvc_path is not None:
        specs.append(
            ModelSpec(
                key="rvc",
                label=_MODEL_LABELS["rvc"],
                path=rvc_path,
            )
        )

    return specs


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch_checkpoint(model_path: Path, device: torch.device) -> Any:
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_checkpoint(
    model_path: Optional[PathLike] = None,
    device: Optional[torch.device] = None,
    model_key: str = "tts",
) -> tuple[torch.nn.Module, float]:
    device = device or default_device()
    resolved_model_path = _resolve_model_path(model_path, model_key=model_key)
    device_key = str(device)
    cache_key = (model_key, resolved_model_path, device_key)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = _load_torch_checkpoint(resolved_model_path, device)
    checkpoint_threshold = DEFAULT_THRESHOLD
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        checkpoint_threshold = float(checkpoint.get("threshold", checkpoint_threshold))
        config = checkpoint.get("config", {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    WhisperEncoderLCNN = _load_deepvoice_model_class()
    model = WhisperEncoderLCNN(
        whisper_size=config.get("whisper_size", "base"),
        freeze_whisper=bool(config.get("freeze_whisper", True)),
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, checkpoint_threshold)
    print(
        f"DeepVoice {_MODEL_LABELS.get(model_key, model_key).upper()} model loaded: "
        f"{resolved_model_path} (threshold={checkpoint_threshold:.2f}, device={device})"
    )
    return _MODEL_CACHE[cache_key]


def audio_to_mel(audio: np.ndarray) -> torch.Tensor:
    audio_tensor = torch.from_numpy(audio).float()
    mel = whisper.log_mel_spectrogram(audio_tensor)
    return mel.unsqueeze(0)


def _normalize_heatmap(value: torch.Tensor) -> torch.Tensor:
    value = value - value.min()
    if float(value.max()) > 0:
        value = value / value.max()
    return value


def _smooth_heatmap(value: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    if kernel_size <= 1:
        return value
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    return F.avg_pool2d(
        value.unsqueeze(0).unsqueeze(0),
        kernel_size=(3, kernel_size),
        stride=1,
        padding=(1, pad),
    ).squeeze(0).squeeze(0)


def _save_heatmap_png(
    mel: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(float(metadata["duration_sec"]), mel.shape[-1] / 100.0)
    extent = [0, duration, 0, mel.shape[0]]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].imshow(mel, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[0].set_title("Whisper log-Mel spectrogram")
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(mel, aspect="auto", origin="lower", extent=extent, cmap="gray_r")
    axes[1].imshow(heatmap, aspect="auto", origin="lower", extent=extent, cmap="jet", alpha=0.55)
    axes[1].set_title(
        "Log-Mel gradient evidence heatmap | "
        f"result={metadata['result']} | fake_prob={metadata['fake_prob']:.4f}"
    )
    axes[1].set_ylabel("Mel bin")

    image = axes[2].imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Normalized log-Mel gradient heatmap")
    axes[2].set_ylabel("Mel bin")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(image, ax=axes[2], fraction=0.025, pad=0.015)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_logmel_heatmap(
    audio_path: str,
    audio: np.ndarray,
    is_fake: int,
    model_path: Optional[PathLike] = None,
    model_key: str = "tts",
    model_label: Optional[str] = None,
) -> dict[str, Any]:
    device = default_device()
    resolved_model_path = _resolve_model_path(model_path, model_key=model_key)
    model, threshold = load_checkpoint(
        resolved_model_path,
        device=device,
        model_key=model_key,
    )
    target_idx = 1 if is_fake else 0
    target_class = "fake" if is_fake else "real"

    original_freeze_whisper = getattr(model, "freeze_whisper", None)
    model.freeze_whisper = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    duration_sec = len(audio) / 16000
    raw_mel = whisper.log_mel_spectrogram(torch.from_numpy(audio).float())
    visible_frames = min(raw_mel.shape[-1], 3000)

    mel = raw_mel.unsqueeze(0).detach().clone().to(device)
    mel.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(mel)
    prob = torch.softmax(logits, dim=1)[0]
    real_prob = float(prob[0].detach().cpu())
    fake_prob = float(prob[1].detach().cpu())
    result = "FAKE" if fake_prob >= threshold else "REAL"

    logits[0, target_idx].backward()

    grad = mel.grad.detach()[0].float().cpu()
    mel_cpu = mel.detach()[0].float().cpu()
    heatmap = (grad * mel_cpu).abs()

    heatmap = heatmap[:, :visible_frames]
    mel_cpu = mel_cpu[:, :visible_frames]
    heatmap = _normalize_heatmap(_smooth_heatmap(heatmap, 7))

    stem = Path(audio_path).stem
    output_path = HEATMAP_DIR / f"{stem}_logmel_heatmap.png"
    metadata = {
        "model_type": "whisper_encoder_lcnn",
        "model_key": model_key,
        "model_label": model_label or _MODEL_LABELS.get(model_key, model_key.upper()),
        "model_path": str(resolved_model_path),
        "visualization": "log-Mel input gradient-based evidence heatmap",
        "threshold": threshold,
        "result": result,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "confidence": fake_prob if result == "FAKE" else real_prob,
        "target_class": target_class,
        "duration_sec": duration_sec,
        "visible_frames": int(visible_frames),
    }

    _save_heatmap_png(mel_cpu.numpy(), heatmap.numpy(), output_path, metadata)
    if original_freeze_whisper is not None:
        model.freeze_whisper = original_freeze_whisper
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(output_path.with_suffix(".heatmap.npy"), heatmap.numpy())
    np.save(output_path.with_suffix(".mel.npy"), mel_cpu.numpy())

    return {
        "url": f"/heatmaps/{output_path.name}",
        "metadata": metadata,
    }


def _predict_mel_with_model(
    mel: torch.Tensor,
    spec: ModelSpec,
    device: torch.device,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    model, checkpoint_threshold = load_checkpoint(
        spec.path,
        device=device,
        model_key=spec.key,
    )
    resolved_threshold = checkpoint_threshold if threshold is None else threshold

    with torch.no_grad():
        logits = model(mel.to(device))
        prob = torch.softmax(logits, dim=1)[0]
        real_prob = float(prob[0].item())
        fake_prob = float(prob[1].item())

    is_fake = 1 if fake_prob >= resolved_threshold else 0
    confidence = fake_prob if is_fake else real_prob
    return {
        "model_key": spec.key,
        "model_label": spec.label,
        "model_path": str(spec.path),
        "is_fake": is_fake,
        "result": "FAKE" if is_fake else "REAL",
        "confidence": round(confidence * 100, 2),
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "threshold": resolved_threshold,
    }


def predict_audio_array_detailed(
    audio: np.ndarray,
    model_path: Optional[PathLike] = None,
    threshold: Optional[float] = None,
    model_key: str = "tts",
    model_label: Optional[str] = None,
) -> dict[str, Any]:
    resolved_model_path = _resolve_model_path(model_path, model_key=model_key)
    spec = ModelSpec(
        key=model_key,
        label=model_label or _MODEL_LABELS.get(model_key, model_key.upper()),
        path=resolved_model_path,
    )
    return _predict_mel_with_model(audio_to_mel(audio), spec, default_device(), threshold)


def predict_audio_array(
    audio: np.ndarray,
    model_path: Optional[PathLike] = None,
    threshold: Optional[float] = None,
) -> tuple[int, float, float]:
    prediction = predict_audio_array_detailed(
        audio,
        model_path=model_path,
        threshold=threshold,
        model_key="tts",
    )
    is_fake = int(prediction["is_fake"])
    confidence = float(prediction["confidence"])
    fake_prob = float(prediction["fake_prob"])
    return is_fake, confidence, fake_prob


def predict_audio_array_all_models(audio: np.ndarray) -> list[dict[str, Any]]:
    device = default_device()
    mel = audio_to_mel(audio)
    return [
        _predict_mel_with_model(mel, spec, device)
        for spec in _available_model_specs()
    ]


def warm_up_models() -> dict[str, Any]:
    start_time = time.time()
    audio = np.zeros(16000, dtype=np.float32)
    librosa.feature.spectral_centroid(y=audio, sr=16000).mean()
    librosa.feature.zero_crossing_rate(audio).mean()
    spec = np.abs(librosa.stft(audio, n_fft=2048))
    librosa.amplitude_to_db(spec + 1e-12, ref=np.max)

    predictions = predict_audio_array_all_models(audio)
    overall = choose_overall_prediction(predictions)
    try:
        generate_logmel_heatmap(
            "__warmup__.wav",
            audio,
            int(overall["is_fake"]),
            model_path=overall["heatmap_model_path"],
            model_key=overall["heatmap_model_key"],
            model_label=overall["heatmap_model_label"],
        )
    finally:
        for suffix in (
            "_logmel_heatmap.png",
            "_logmel_heatmap.json",
            "_logmel_heatmap.heatmap.npy",
            "_logmel_heatmap.mel.npy",
        ):
            path = HEATMAP_DIR / f"__warmup__{suffix}"
            if path.exists():
                path.unlink()

    return {
        "models": [prediction["model_key"] for prediction in predictions],
        "elapsed": round(time.time() - start_time, 3),
    }


def choose_overall_prediction(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise RuntimeError("No DeepVoice model checkpoints are available.")

    tts_prediction = next(
        (prediction for prediction in predictions if prediction["model_key"] == "tts"),
        None,
    )
    rvc_prediction = next(
        (prediction for prediction in predictions if prediction["model_key"] == "rvc"),
        None,
    )

    tts_detected = bool(tts_prediction and tts_prediction["is_fake"])
    rvc_detected = bool(rvc_prediction and rvc_prediction["is_fake"])
    heatmap_source = max(predictions, key=lambda prediction: prediction["fake_prob"])

    if tts_detected and rvc_detected:
        return {
            **heatmap_source,
            "is_fake": 1,
            "confidence": round(float(heatmap_source["fake_prob"]) * 100, 2),
            "fake_prob": float(heatmap_source["fake_prob"]),
            "decision_model": "combined",
            "final_label": "가짜 음성",
            "detail_type": "TTS/RVC 복합 유형",
            "heatmap_model_key": heatmap_source["model_key"],
            "heatmap_model_label": heatmap_source["model_label"],
            "heatmap_model_path": heatmap_source["model_path"],
        }

    if tts_detected:
        return {
            **tts_prediction,
            "decision_model": "tts",
            "final_label": "가짜 음성",
            "detail_type": "TTS 합성음성",
            "heatmap_model_key": tts_prediction["model_key"],
            "heatmap_model_label": tts_prediction["model_label"],
            "heatmap_model_path": tts_prediction["model_path"],
        }

    if rvc_detected:
        return {
            **rvc_prediction,
            "decision_model": "rvc",
            "final_label": "가짜 음성",
            "detail_type": "RVC 변환음성",
            "heatmap_model_key": rvc_prediction["model_key"],
            "heatmap_model_label": rvc_prediction["model_label"],
            "heatmap_model_path": rvc_prediction["model_path"],
        }

    real_source = min(predictions, key=lambda prediction: prediction["real_prob"])
    return {
        **real_source,
        "is_fake": 0,
        "confidence": round(float(real_source["real_prob"]) * 100, 2),
        "fake_prob": float(real_source["fake_prob"]),
        "decision_model": "real",
        "final_label": "실제 음성",
        "detail_type": "해당 없음",
        "heatmap_model_key": real_source["model_key"],
        "heatmap_model_label": real_source["model_label"],
        "heatmap_model_path": real_source["model_path"],
    }


def extract_visual_features(audio_path: str) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30.0)
    duration = librosa.get_duration(y=y, sr=sr)

    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    zero_crossing_rate = float(librosa.feature.zero_crossing_rate(y).mean())

    spec = np.abs(librosa.stft(y, n_fft=2048))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    max_idx = min(len(freq_bins) - 1, int(8000 / (sr / 2048)))
    target_indices = np.linspace(0, max_idx, 64, dtype=int)
    freq_data = spec_db[target_indices, :].mean(axis=1).tolist()

    return {
        "audio": y,
        "duration": duration,
        "sample_rate": sr,
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zero_crossing_rate,
        "freq_data": freq_data,
    }


def predict_audio_file(
    audio_path: str,
    model_path: Optional[PathLike] = None,
    threshold: Optional[float] = None,
) -> tuple[int, float, float]:
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return predict_audio_array(audio, model_path=model_path, threshold=threshold)


def analyze_audio_sync(file_path: str) -> dict[str, Any]:
    start_time = time.time()
    features = extract_visual_features(file_path)
    model_results = predict_audio_array_all_models(features["audio"])
    overall = choose_overall_prediction(model_results)
    is_fake = int(overall["is_fake"])
    confidence = float(overall["confidence"])
    fake_prob = float(overall["fake_prob"])
    heatmap = generate_logmel_heatmap(
        file_path,
        features["audio"],
        is_fake,
        model_path=overall["heatmap_model_path"],
        model_key=overall["heatmap_model_key"],
        model_label=overall["heatmap_model_label"],
    )
    processing_time = round(time.time() - start_time, 3)

    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "processing_time": processing_time,
        "duration": features["duration"],
        "sample_rate": features["sample_rate"],
        "spectral_centroid": features["spectral_centroid"],
        "zero_crossing_rate": features["zero_crossing_rate"],
        "freq_data": features["freq_data"],
        "fake_prob": round(fake_prob * 100, 2),
        "decision_model": overall["decision_model"],
        "final_label": overall["final_label"],
        "detail_type": overall["detail_type"],
        "model_results": model_results,
        "heatmap_url": heatmap["url"],
        "heatmap_metadata": heatmap["metadata"],
    }
