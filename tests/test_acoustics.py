import os
import sys
import tempfile
import unittest
import wave
from datetime import datetime, timedelta
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustics import audio_models as am  # noqa: E402
from acoustics.feeding_metrics import average, calculate_metrics, parse_predictions  # noqa: E402

SAMPLE_WAV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "acoustics", "test_data", "recording_01_bite_0.wav")


class FakeSession:
    """Stands in for onnxruntime.InferenceSession, returning a fixed model output."""

    def __init__(self, output):
        self._output = output

    def get_inputs(self):
        return [mock.Mock(name="input")]

    def run(self, names, feeds):
        self.last_input = next(iter(feeds.values()))
        return [self._output]


def patch_session(output):
    return mock.patch("onnxruntime.InferenceSession", lambda *a, **k: FakeSession(output))


class AudioPlumbingTests(unittest.TestCase):
    def test_raw_pcm_becomes_a_wav_with_the_microphone_format(self):
        path = os.path.join(tempfile.mkdtemp(), "out.wav")
        raw = (np.arange(2000, dtype=np.int16)).tobytes()
        am.save_raw_to_wav(raw, path)
        with wave.open(path, "rb") as wf:
            self.assertEqual((wf.getnchannels(), wf.getsampwidth(), wf.getframerate()), (1, 2, 22500))
            self.assertEqual(wf.getnframes(), 2000)

    def test_sample_rate_argument_is_honoured(self):
        path = os.path.join(tempfile.mkdtemp(), "out.wav")
        am.save_raw_to_wav(b"\x00\x00" * 10, path, sample_rate=16000)
        with wave.open(path, "rb") as wf:
            self.assertEqual(wf.getframerate(), 16000)

    def test_sample_recording_is_in_the_expected_format(self):
        with wave.open(SAMPLE_WAV, "rb") as wf:
            self.assertEqual((wf.getnchannels(), wf.getsampwidth()), (1, 2))


class FeatureTests(unittest.TestCase):
    def test_mfcc_row_has_13_by_128_values_and_pads_short_clips_with_zeros(self):
        feat = am.process_audio(SAMPLE_WAV)
        self.assertEqual(feat.shape, (1, 13 * 128))
        self.assertEqual(feat.dtype, np.float32)
        grid = feat.reshape(13, 128)
        self.assertTrue(np.all(grid[:, -5:] == 0))            # the 0.6 s clip is shorter than 128 frames
        self.assertFalse(np.all(grid[:, :5] == 0))

    def test_spectrogram_image_has_the_size_the_cnn_expects(self):
        mel = np.random.RandomState(0).rand(90, 300) * -40
        img = am.scale_melspec(mel)
        self.assertEqual(img.shape, (200, 90))                # PIL size (90, 200) -> array (rows, cols)
        self.assertEqual(img.dtype, np.uint8)


class BiteChewTests(unittest.TestCase):
    def test_class_index_maps_to_label(self):
        for index, label in ((0, "chew"), (1, "bite"), (2, "chew-bite")):
            with patch_session(np.array([index])):
                self.assertEqual(am.classify_bite_chew(SAMPLE_WAV, "fake.onnx"), label)

    def test_unknown_index_is_labelled_unknown(self):
        with patch_session(np.array([7])):
            self.assertEqual(am.classify_bite_chew(SAMPLE_WAV, "fake.onnx"), "unknown")

    def test_model_gets_one_feature_row(self):
        session = FakeSession(np.array([1]))
        with mock.patch("onnxruntime.InferenceSession", lambda *a, **k: session):
            am.classify_bite_chew(SAMPLE_WAV, "fake.onnx")
        self.assertEqual(session.last_input.shape, (1, 13 * 128))

    def test_missing_model_file_is_a_clear_error(self):
        missing = os.path.join(tempfile.mkdtemp(), "nope.onnx")
        with mock.patch.dict(os.environ), mock.patch.object(am, "DEFAULT_BITE_CHEW_MODEL", missing):
            os.environ.pop("BITE_CHEW_MODEL_PATH", None)
            with self.assertRaises(FileNotFoundError) as ctx:
                am.classify_bite_chew(SAMPLE_WAV)
        self.assertIn("BITE_CHEW_MODEL_PATH", str(ctx.exception))

    def test_environment_variable_overrides_the_shipped_model(self):
        with mock.patch.dict(os.environ, {"BITE_CHEW_MODEL_PATH": SAMPLE_WAV}):   # any existing file
            self.assertEqual(am._resolve_model(None, "BITE_CHEW_MODEL_PATH", "unused.onnx"), SAMPLE_WAV)

    def test_explicit_path_wins_even_if_the_file_is_not_checked(self):
        self.assertEqual(am._resolve_model("explicit.onnx", "BITE_CHEW_MODEL_PATH", "d.onnx"), "explicit.onnx")

    def test_shipped_model_locations_are_inside_the_repo_models_folder(self):
        for p in (am.DEFAULT_BITE_CHEW_MODEL, am.DEFAULT_DISTRESS_MODEL):
            self.assertEqual(os.path.basename(os.path.dirname(p)), "acoustics")
            self.assertEqual(os.path.basename(os.path.dirname(os.path.dirname(p))), "models")


class DistressTests(unittest.TestCase):
    def test_high_probability_is_hfc_with_that_confidence(self):
        with patch_session(np.array([[0.93]])):
            call, conf = am.detect_distress(SAMPLE_WAV, "fake.onnx")
        self.assertEqual(call, am.Predictions.HFC)
        self.assertAlmostEqual(conf, 0.93, places=2)

    def test_low_probability_is_lfc_with_inverted_confidence(self):
        with patch_session(np.array([[0.2]])):
            call, conf = am.detect_distress(SAMPLE_WAV, "fake.onnx")
        self.assertEqual(call, am.Predictions.LFC)
        self.assertAlmostEqual(conf, 0.8, places=2)

    def test_exactly_half_counts_as_hfc(self):
        with patch_session(np.array([[0.5]])):
            self.assertEqual(am.detect_distress(SAMPLE_WAV, "fake.onnx")[0], am.Predictions.HFC)

    def test_failure_returns_none_none_instead_of_raising(self):
        self.assertEqual(am.detect_distress("no_such_file.wav", "fake.onnx"), (None, None))

    def test_input_tensor_shape_is_batch_height_width_rgb(self):
        session = FakeSession(np.array([[0.1]]))
        with mock.patch("onnxruntime.InferenceSession", lambda *a, **k: session):
            am.detect_distress(SAMPLE_WAV, "fake.onnx")
        self.assertEqual(session.last_input.shape, (1, 200, 90, 3))

    def test_missing_model_file_is_a_clear_error(self):
        missing = os.path.join(tempfile.mkdtemp(), "nope.onnx")
        with mock.patch.dict(os.environ), mock.patch.object(am, "DEFAULT_DISTRESS_MODEL", missing):
            os.environ.pop("DISTRESS_MODEL_PATH", None)
            with self.assertRaises(FileNotFoundError) as ctx:
                am.detect_distress(SAMPLE_WAV)
        self.assertIn("DISTRESS_MODEL_PATH", str(ctx.exception))


class FeedingMetricsTests(unittest.TestCase):
    T0 = datetime(2026, 10, 16, 8, 0, 0)

    def events(self, minutes, label="chew"):
        return [(self.T0 + timedelta(minutes=m), label) for m in minutes]

    def test_a_long_silence_splits_meals(self):
        preds = self.events([0, 1, 2, 3]) + self.events([20, 21, 22])     # 17 min gap > 5 min
        sessions = parse_predictions(preds)
        self.assertEqual([len(s) for s in sessions], [4, 3])

    def test_non_feeding_labels_are_ignored(self):
        preds = self.events([0, 1]) + [(self.T0 + timedelta(minutes=2), "noise")] + self.events([3])
        self.assertEqual(len(parse_predictions(preds)[0]), 3)

    def test_metrics_for_two_meals(self):
        preds = self.events([0, 1, 2, 3]) + self.events([63, 64, 65])     # meals of 3 and 2 minutes, 60 min apart
        m = calculate_metrics(preds)
        self.assertEqual(m["Feeding Frequency (FF)"], 2)
        self.assertAlmostEqual(m["Feeding Time (FT)"], 5.0)
        self.assertEqual(m["Meal Durations (MD)"], [3.0, 2.0])
        self.assertAlmostEqual(m["Average Feeding Time (AFT)"], 2.5)
        self.assertAlmostEqual(m["Inter-Meal Intervals (IMI)"][0], 1.0, places=3)   # meal 1 ends at 3 min, meal 2 starts at 63
        self.assertEqual(m["Total Chews/Bites Per Day"], 7)
        self.assertAlmostEqual(m["Feeding Rates (FR)"][0], 4 / 3, places=3)

    def test_no_predictions_gives_zeros_without_dividing_by_zero(self):
        m = calculate_metrics([])
        self.assertEqual(m["Feeding Frequency (FF)"], 0)
        self.assertEqual(m["Average Feeding Time (AFT)"], 0)
        self.assertEqual(m["Total Chews/Bites Per Day"], 0)

    def test_average_of_nothing_is_zero(self):
        self.assertEqual(average([]), 0)
        self.assertEqual(average([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
