# Acoustic detection

Cow audio (from a wearable ESP32 microphone) -> bite/chew and distress-call detection.
Code adapted from the team's `livestock-monitoring` repo
(https://github.com/shashi007-hub/livestock-monitoring, commit `a4a2c94`).

## What is here

| Path | What it is |
|---|---|
| `audio_models.py` | bite/chew classifier (MFCC -> XGBoost ONNX), distress-call classifier (mel spectrogram -> CNN ONNX), WAV saving. Extracted from the backend's `mapping.py` with the database, SMS and logging code removed. |
| `feeding_metrics.py` | meals, feeding time, rates and intervals from bite/chew predictions (from the backend's `cron.py`, calculation only) |
| `firmware/microphone/` | MicroPython for the ESP32 + INMP441 mic: records 5 s on a button press and streams it over MQTT. `boot.py` has placeholders for the Wi-Fi name and password. |
| `test_data/` | `recording_01_bite_0.wav` plus 5 HFC and 5 LFC clips (`hfc/`, `lfc/`) |
| `../models/acoustics/` | the two ONNX models: `xgb_model.onnx` (0.4 MB), `cow_model.onnx` (62 MB) |

## Use

```python
from acoustics.audio_models import classify_bite_chew, detect_distress
classify_bite_chew("clip.wav")        # "chew" | "bite" | "chew-bite"
detect_distress("clip.wav")           # (Predictions.HFC, 1.0)   HFC = high-frequency call = distress
```

The models are found automatically in `models/acoustics/`. To use other files, pass `model_path=...` or set
`BITE_CHEW_MODEL_PATH` / `DISTRESS_MODEL_PATH`. Install with `pip install -r acoustics/requirements.txt`, test with
`pytest tests/test_acoustics.py tests/test_acoustics_models.py`.

Audio format from the microphone: mono, 16-bit PCM, **22,500 Hz**, 5 s clips. Other files are resampled on load.

## How well does it work? (one run, October 2026)

Distress model on the 10 clips in `test_data/` (from the team's local HFC/LFC recordings; the source README points to the
Kaggle cattle bioacoustic dataset, so check that dataset's licence before making this repo public):

| Expected | Clip | Distress model | Confidence | Bite/chew model |
|---|---|---|---|---|
| HFC | `1083.HFC_11.00_1` | HFC | 1.00 | chew |
| HFC | `1111.HFC_08.45_1` | HFC | 1.00 | chew-bite |
| HFC | `1117.HFC_09.15_1` | HFC | 1.00 | chew-bite |
| HFC | `1123.HFC_09.45_2` | HFC | 1.00 | chew-bite |
| HFC | `1125.HFC_10.00_1` | **LFC** (wrong) | 1.00 | chew-bite |
| LFC | `329.0.LFC_augmented_329.0` | LFC | 1.00 | chew-bite |
| LFC | `369.0.LFC_augmented_369.0` | LFC | 1.00 | chew |
| LFC | `38.0.LFC_augmented_38.0` | **HFC** (wrong) | 1.00 | chew-bite |
| LFC | `689.0.LFC_augmented_689.0` | LFC | 1.00 | chew |
| LFC | `758.0.LFC_augmented_758.0` | **HFC** (wrong) | 1.00 | chew |

**7 of 10 correct.** What to keep in mind:
- Ten clips is a smoke test, not an accuracy figure. The tests only check that every clip gets a valid answer.
- The confidence is always about 1.0, so it cannot tell a sure answer from a wrong one.
- Earlier checks also had it call a bite recording and a 440 Hz tone "HFC", so it may lean towards distress.
- The HFC and LFC recordings come from different setups (44-48 kHz vs 22 kHz), which can teach a model the recording
  instead of the sound. The LFC clips are named "augmented", and may overlap the model's training data.
- The bite/chew model has no "background" class: silence and noise still come out as "chew". The one clip with a
  known label (`recording_01_bite_0.wav`) is correctly called "bite" (99.8%). It was not run on labelled chew clips.

Treat both models as **unvalidated** until they are tested on labelled clips they were not trained on.

## Not included

The backend that listens on MQTT, stores results and sends SMS, and the camera/accelerometer pipelines.
The team's Drive folder `models` also holds `yolo11n.onnx`, `vit_cow_disease_model.onnx`, `Scalar_lamness.pkl` and
`trained_modelnew.pkl` for those other detectors. Don't load `.pkl` files from sources you don't trust: pickle
files can run code when loaded.
