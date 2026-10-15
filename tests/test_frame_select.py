import os
import tempfile
import unittest

import cv2
import numpy as np

from frame_extraction import Config, Detection, assess_frame, select_best_frames

W, H, FPS, N = 320, 240, 10, 30


def make_video(path):
    """30 frames of noise texture (sharp) so Laplacian variance is high; frame 20 is blurred."""
    rng = np.random.RandomState(0)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        frame = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
        if i == 20:
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        out.write(frame)
    out.release()


class FakeDetector:
    """Scripted detections by call order (one call per sampled frame)."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, frame):
        dets = self.script.get(self.calls, [])
        self.calls += 1
        return dets


GOOD = Detection(60, 60, 260, 180, 0.9)           # 200x120, aspect 1.67, well inside
CUT_OFF = Detection(0, 60, 200, 180, 0.9)         # touches left edge
TALL = Detection(100, 20, 180, 220, 0.9)          # aspect 0.4, not side-on
SMALL = Detection(10, 10, 60, 40, 0.9)


class AssessTests(unittest.TestCase):
    cfg = Config()
    frame = np.zeros((H, W, 3), np.uint8)

    def test_good_passes(self):
        self.assertIsNotNone(assess_frame(self.frame, [GOOD], self.cfg))

    def test_cut_off_rejected(self):
        self.assertIsNone(assess_frame(self.frame, [CUT_OFF], self.cfg))

    def test_low_conf_rejected(self):
        self.assertIsNone(assess_frame(self.frame, [Detection(60, 60, 260, 180, 0.2)], self.cfg))

    def test_small_rejected(self):
        self.assertIsNone(assess_frame(self.frame, [SMALL], self.cfg))

    def test_occluded_by_other_cow_rejected(self):
        other = Detection(100, 60, 240, 180, 0.8)
        self.assertIsNone(assess_frame(self.frame, [GOOD, other], self.cfg))

    def test_far_away_second_cow_ok(self):
        other = Detection(0, 0, 20, 20, 0.8)
        self.assertIsNotNone(assess_frame(self.frame, [GOOD, other], self.cfg))

    def test_tall_box_scores_lower_side_than_good(self):
        good = assess_frame(self.frame, [GOOD], self.cfg)
        tall = assess_frame(self.frame, [Detection(60, 10, 140, 230, 0.9)], Config(min_area_frac=0.0))
        self.assertEqual(good.side_score, 1.0)
        self.assertLess(tall.side_score, 0.5)


class SelectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "clip.mp4")
        make_video(self.video)

    def test_sampling_and_blurry_frame_not_chosen(self):
        # sample_fps == FPS -> every frame is sampled, call index == frame index
        script = {i: [GOOD] for i in range(N)}
        det = FakeDetector(script)
        res = select_best_frames(self.video, det, Config(sample_fps=FPS))
        self.assertEqual(det.calls, N)
        self.assertEqual(len(res), 1)
        self.assertNotEqual(res[0][0].frame_index, 20)  # the blurred one

    def test_picks_the_only_valid_frame(self):
        script = {i: [CUT_OFF] for i in range(N)}
        script[7] = [GOOD]
        res = select_best_frames(self.video, FakeDetector(script), Config(sample_fps=FPS))
        self.assertEqual(res[0][0].frame_index, 7)

    def test_none_valid_returns_empty(self):
        res = select_best_frames(self.video, FakeDetector({}), Config(sample_fps=FPS))
        self.assertEqual(res, [])

    def test_top_k(self):
        script = {i: [GOOD] for i in range(N)}
        res = select_best_frames(self.video, FakeDetector(script), Config(sample_fps=FPS), top_k=3)
        self.assertEqual(len(res), 3)
        scores = [c.score for c, _ in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_missing_video(self):
        with self.assertRaises(FileNotFoundError):
            select_best_frames(os.path.join(self.tmp, "nope.mp4"), FakeDetector({}))


if __name__ == "__main__":
    unittest.main()
