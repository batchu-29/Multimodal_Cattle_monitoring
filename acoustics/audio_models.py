"""Acoustic detection from a short cow audio clip: bite/chew and distress calls.

Extracted from livestock-monitoring (github.com/shashi007-hub/livestock-monitoring,
backend/worker/app/mapping.py, commit a4a2c94). The feature extraction and model-calling
logic is the same; what was removed is everything tied to that backend (the database,
SMS alerts, the multiprocess logger, and the camera and lameness pipelines).

    from acoustics.audio_models import classify_bite_chew, detect_distress
    label = classify_bite_chew("clip.wav")            # "chew" | "bite" | "chew-bite"
    call, confidence = detect_distress("clip.wav")    # (Predictions.HFC | LFC, 0-1)

The two ONNX models are in models/acoustics/ and are found automatically. To use other files, pass
model_path=... or set these environment variables:
    BITE_CHEW_MODEL_PATH   MFCC features -> class 0 chew / 1 bite / 2 chew-bite   (XGBoost)
    DISTRESS_MODEL_PATH    mel spectrogram image -> P(high-frequency call, HFC)   (CNN)

Audio format from the ESP32 microphone: mono, 16-bit, 22,500 Hz, 5 s clips.
Heavy imports (librosa, onnxruntime, PIL) happen inside functions so this module imports fast.
"""
from __future__ import annotations

import logging
import os
import wave
from enum import Enum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("acoustics")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "acoustics")
DEFAULT_BITE_CHEW_MODEL = os.path.join(MODELS_DIR, "xgb_model.onnx")
DEFAULT_DISTRESS_MODEL = os.path.join(MODELS_DIR, "cow_model.onnx")

LABELS = {0: "chew", 1: "bite", 2: "chew-bite"}
MIC_SAMPLE_RATE = 22500          # must match the ESP32 recording config (firmware/microphone/main.py)
MFCC_FRAMES = 128                # the bite/chew model expects 13 x 128 MFCC values, padded or cut


class Predictions(Enum):
    HFC = "HFC"      # high-frequency call: the distress class
    LFC = "LFC"      # low-frequency call


# ---------------------------------------------------------------- bite / chew (MFCC + XGBoost ONNX)
def extract_mfcc(y, sr, n_mfcc=13, hop_length=256, n_fft=1024):
    import librosa

    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfcc_feat.astype(np.float32)


def process_audio(filepath, sr=22050):
    """WAV file -> (1, 13 * 128) float32 feature row, zero-padded or cut to 128 frames."""
    import librosa

    y, _ = librosa.load(filepath, sr=sr)
    mfcc = extract_mfcc(y, sr)
    if mfcc.shape[1] < MFCC_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MFCC_FRAMES - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MFCC_FRAMES]
    feature = mfcc.flatten().astype(np.float32)
    return np.expand_dims(feature, axis=0)


def run_inference(onnx_model_path, audio_path):
    """Raw class index from the bite/chew model."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: process_audio(audio_path)})
    return outputs[0][0]


def _resolve_model(model_path: Optional[str], env_var: str, default: str) -> str:
    """An explicit path wins, then the environment variable, then the model shipped in the repo.
    A missing shipped/env model gets a clear error instead of a cryptic runtime one."""
    if model_path:
        return model_path
    path = os.getenv(env_var) or default
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model file not found: {path} (pass model_path or set {env_var})")
    return path


def classify_bite_chew(audio_path: str, model_path: Optional[str] = None) -> str:
    model_path = _resolve_model(model_path, "BITE_CHEW_MODEL_PATH", DEFAULT_BITE_CHEW_MODEL)
    return LABELS.get(int(run_inference(model_path, audio_path)), "unknown")


# ---------------------------------------------------------------- distress call (mel spectrogram CNN ONNX)
def scale_melspec(mel_spec):
    """Normalise a log-mel spectrogram and resize it to the image the CNN was trained on."""
    import librosa
    from PIL import Image

    target_size = (90, 200)
    norm_spec = librosa.util.normalize(mel_spec)
    norm_spec = (norm_spec * 255).astype(np.uint8)
    img = Image.fromarray(norm_spec).resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)


def predict_from_wav(model_path, wav_path) -> Tuple[Optional[Predictions], Optional[float]]:
    """(HFC | LFC, confidence), or (None, None) if anything fails (the failure is logged)."""
    try:
        import librosa
        import onnxruntime as ort

        y, sr = librosa.load(wav_path)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
        scaled = scale_melspec(librosa.power_to_db(mel_spec))
        # shape it to the model input: (1, H, W, 3)
        tensor = np.expand_dims(np.repeat(scaled[..., np.newaxis], 3, axis=-1), axis=0).astype(np.float32)

        session = ort.InferenceSession(model_path)
        prediction = session.run(None, {session.get_inputs()[0].name: tensor})[0]

        score = float(prediction[0][0])                       # probability of class 1 (HFC)
        pred_class = 1 if score >= 0.5 else 0
        confidence = score if pred_class == 1 else 1 - score
        logger.info("distress prediction: class %s, confidence %.4f", pred_class, confidence)
        return (Predictions.HFC if pred_class == 1 else Predictions.LFC), confidence
    except Exception:
        logger.error("error during distress model inference", exc_info=True)
        return None, None


def detect_distress(audio_path: str, model_path: Optional[str] = None) -> Tuple[Optional[Predictions], Optional[float]]:
    model_path = _resolve_model(model_path, "DISTRESS_MODEL_PATH", DEFAULT_DISTRESS_MODEL)
    return predict_from_wav(model_path, audio_path)


# ---------------------------------------------------------------- audio plumbing
def save_raw_to_wav(raw_bytes: bytes, output_wav_path: str, sample_rate: int = MIC_SAMPLE_RATE) -> None:
    """Raw 16-bit mono PCM from the microphone -> WAV file. (The original ignored its
    sample_rate argument and always used 22,500 Hz; that is now the default and is honoured.)"""
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
