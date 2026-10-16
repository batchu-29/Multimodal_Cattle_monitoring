"""Runs the two REAL ONNX models (models/acoustics/) on the recordings in acoustics/test_data/.

These check that the models load, accept our features and return valid answers. They do not
assert accuracy: the distress model is known to make mistakes (see acoustics/README.md).
"""
import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustics import audio_models as am  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acoustics", "test_data")
BITE = os.path.join(DATA, "recording_01_bite_0.wav")
HFC = sorted(glob.glob(os.path.join(DATA, "hfc", "*.wav")))
LFC = sorted(glob.glob(os.path.join(DATA, "lfc", "*.wav")))


@unittest.skipUnless(os.path.isfile(am.DEFAULT_BITE_CHEW_MODEL), "bite/chew model not available")
class RealBiteChewModel(unittest.TestCase):
    def test_labels_the_bite_recording_a_bite(self):
        self.assertEqual(am.classify_bite_chew(BITE), "bite")

    def test_every_test_clip_gets_one_of_the_three_labels(self):
        for path in HFC + LFC:
            self.assertIn(am.classify_bite_chew(path), set(am.LABELS.values()), path)


@unittest.skipUnless(os.path.isfile(am.DEFAULT_DISTRESS_MODEL), "distress model not available")
class RealDistressModel(unittest.TestCase):
    def test_there_are_five_clips_of_each_call_type(self):
        self.assertEqual((len(HFC), len(LFC)), (5, 5))

    def test_every_clip_gets_a_valid_prediction(self):
        for path in HFC + LFC:
            call, conf = am.detect_distress(path)
            self.assertIn(call, (am.Predictions.HFC, am.Predictions.LFC), path)   # None = input rejected
            self.assertTrue(0.5 <= conf <= 1.0, path)


if __name__ == "__main__":
    unittest.main()
