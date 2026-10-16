import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_estimation import SUPERANIMAL_QUADRUPED, SuperAnimalBackend  # noqa: E402


class RecordingBackend(SuperAnimalBackend):
    """Replaces the heavy model with a stub that tags each pose with its frame index."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def _infer(self, frames):
        self.seen.append(len(frames))
        return [{"nose": (float(i), 0.0, 1.0)} for i in range(len(frames))]


FRAME = np.zeros((10, 10, 3), np.uint8)


class PaddingTests(unittest.TestCase):
    def test_single_frame_is_sent_twice_and_one_pose_returned(self):
        b = RecordingBackend()
        poses = b.estimate_frames([FRAME])
        self.assertEqual(b.seen, [2])              # the model never sees a one-frame video
        self.assertEqual(len(poses), 1)

    def test_several_frames_pass_through_unchanged(self):
        b = RecordingBackend()
        poses = b.estimate_frames([FRAME] * 5)
        self.assertEqual(b.seen, [5])
        self.assertEqual([p["nose"][0] for p in poses], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_no_frames_returns_nothing_without_calling_the_model(self):
        b = RecordingBackend()
        self.assertEqual(b.estimate_frames([]), [])
        self.assertEqual(b.seen, [])


class MappingTests(unittest.TestCase):
    def test_canonical_names_are_unique_and_cover_all_four_legs(self):
        names = list(SUPERANIMAL_QUADRUPED.values())
        self.assertEqual(len(names), len(set(names)))
        for leg in ("front_left", "front_right", "hind_left", "hind_right"):
            for part in ("thigh", "knee", "paw"):
                self.assertIn(f"{leg}_{part}", names)


if __name__ == "__main__":
    unittest.main()
