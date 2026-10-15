import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activity_monitoring import (  # noqa: E402
    LYING, SITTING, STANDING, UNKNOWN, Config, classify_frame, classify_pose,
)


def make_pose(front_drop, hind_drop, scale=1.0, mirror=False, conf=0.9, x0=200, y0=150):
    """Side-view cow, trunk length 100 units. `*_drop` = paw below thigh in trunk lengths."""
    s = -1 if mirror else 1

    def P(dx, dy, c=conf):
        return (x0 + s * dx * scale, y0 + dy * scale, c)

    return {
        "front_left_thigh": P(50, 0), "front_right_thigh": P(46, 0),
        "hind_left_thigh": P(-50, 0), "hind_right_thigh": P(-46, 0),
        "front_left_paw": P(52, front_drop * 100), "front_right_paw": P(44, front_drop * 100),
        "hind_left_paw": P(-52, hind_drop * 100), "hind_right_paw": P(-44, hind_drop * 100),
    }


class ClassifyTests(unittest.TestCase):
    def test_standing(self):
        r = classify_pose(make_pose(0.7, 0.7))
        self.assertEqual(r.label, STANDING)
        self.assertGreater(r.confidence, 0.8)

    def test_lying(self):
        self.assertEqual(classify_pose(make_pose(0.05, 0.05)).label, LYING)

    def test_sitting_rear_down(self):
        self.assertEqual(classify_pose(make_pose(0.7, 0.05)).label, SITTING)

    def test_front_down_hind_up_is_sitting_transition(self):
        self.assertEqual(classify_pose(make_pose(0.05, 0.7)).label, SITTING)

    def test_camera_distance_does_not_matter(self):
        for scale in (0.3, 1, 4):
            self.assertEqual(classify_pose(make_pose(0.7, 0.7, scale=scale)).label, STANDING, scale)
            self.assertEqual(classify_pose(make_pose(0.05, 0.05, scale=scale)).label, LYING, scale)

    def test_direction_does_not_matter(self):
        for mirror in (False, True):
            self.assertEqual(classify_pose(make_pose(0.7, 0.7, mirror=mirror)).label, STANDING)
            self.assertEqual(classify_pose(make_pose(0.7, 0.05, mirror=mirror)).label, SITTING)

    def test_borderline_is_low_confidence(self):
        near = classify_pose(make_pose(0.36, 0.36))
        clear = classify_pose(make_pose(0.8, 0.8))
        self.assertLess(near.confidence, 0.2)
        self.assertGreater(clear.confidence, near.confidence)

    def test_low_keypoint_confidence_lowers_confidence(self):
        self.assertLess(classify_pose(make_pose(0.8, 0.8, conf=0.4)).confidence,
                        classify_pose(make_pose(0.8, 0.8, conf=0.95)).confidence)

    def test_unreliable_keypoints_are_ignored(self):
        pose = make_pose(0.7, 0.7)
        pose["front_left_paw"] = (0, 0, 0.05)       # garbage but low confidence: skipped
        r = classify_pose(pose)
        self.assertEqual(r.label, STANDING)

    def test_one_good_leg_per_end_is_enough(self):
        pose = make_pose(0.7, 0.7)
        for n in ("front_right_thigh", "front_right_paw", "hind_right_thigh", "hind_right_paw"):
            del pose[n]
        self.assertEqual(classify_pose(pose).label, STANDING)

    def test_no_hind_legs_is_unknown(self):
        pose = {k: v for k, v in make_pose(0.7, 0.7).items() if not k.startswith("hind")}
        r = classify_pose(pose)
        self.assertEqual(r.label, UNKNOWN)
        self.assertEqual(r.confidence, 0.0)

    def test_paws_hidden_is_unknown(self):
        pose = {k: v for k, v in make_pose(0.7, 0.7).items() if "paw" not in k}
        self.assertEqual(classify_pose(pose).label, UNKNOWN)

    def test_tiny_cow_is_unknown(self):
        self.assertEqual(classify_pose(make_pose(0.7, 0.7, scale=0.1)).label, UNKNOWN)

    def test_empty_pose_is_unknown(self):
        self.assertEqual(classify_pose({}).label, UNKNOWN)

    def test_threshold_is_configurable(self):
        pose = make_pose(0.3, 0.3)
        self.assertEqual(classify_pose(pose).label, LYING)
        self.assertEqual(classify_pose(pose, Config(extended_at=0.25)).label, STANDING)

    def test_classify_frame_uses_injected_pose_fn(self):
        seen = []
        frame = np.zeros((10, 10, 3), np.uint8)
        r = classify_frame(frame, pose_fn=lambda f: seen.append(f) or make_pose(0.05, 0.05))
        self.assertEqual(r.label, LYING)
        self.assertIs(seen[0], frame)


if __name__ == "__main__":
    unittest.main()
