import base64
import json
import os
import sys
import unittest
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gait_detection as gd  # noqa: E402

NOW = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)


def frame(shade):
    img = np.full((120, 200, 3), shade, np.uint8)
    cv2.putText(img, str(shade), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


# ranked best-first; times are deliberately NOT in order so we can check the time-sorting
RANKED = [(frame(10), 3.0), (frame(20), 1.0), (frame(30), 2.0), (frame(40), 4.0), (frame(50), 0.5)]


def fake_pick(video, n):
    return RANKED[:n]


def fake_pose(frames):
    return [{"nose": (10.123, 20.456, 0.9), "tail_base": (100, 50, 0.2), "back_middle": (50, 40, 0.5)}
            for _ in frames]


GOOD = json.dumps({"image_quality_ok": True, "gait_score": 2, "summary": "slight head nod,\nshort rear stride"})


class FakeWriter:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


class GateTests(unittest.TestCase):
    def test_good(self):
        self.assertEqual(gd.parse_response(GOOD), (2, "slight head nod,\nshort rear stride"))

    def test_rejections_give_no_score(self):
        def resp(**kw):
            d = json.loads(GOOD)
            d.update(kw)
            return json.dumps(d)

        for bad in (resp(image_quality_ok=False), resp(image_quality_ok="true"), resp(gait_score=0),
                    resp(gait_score=6), resp(gait_score=2.5), resp(gait_score=True), resp(gait_score="2"),
                    "not json", "[]", None):
            score, reason = gd.parse_response(bad)
            self.assertIsNone(score, bad)
            self.assertTrue(reason)

    def test_one_line(self):
        line = gd.one_line(3, "a\n\n b   c")
        self.assertEqual(line, "score=3 | a b c")
        self.assertNotIn("\n", line)
        self.assertTrue(gd.one_line(None, "bad image").startswith("NO SCORE | "))
        self.assertLessEqual(len(gd.one_line(1, "x" * 1000)), 300)


class PromptTests(unittest.TestCase):
    def test_pose_summary_drops_low_confidence_and_rounds(self):
        s = gd.pose_summary(fake_pose([0])[0])
        self.assertEqual(s, {"nose": [10.1, 20.5, 0.9], "back_middle": [50, 40, 0.5]})

    def test_contents_are_interleaved_in_given_order(self):
        frames = [RANKED[1], RANKED[2]]
        contents = gd.build_contents(frames, fake_pose([0, 0]))
        kinds = ["img" if isinstance(c, bytes) else "txt" for c in contents]
        self.assertEqual(kinds, ["txt", "txt", "img", "txt", "img"])
        self.assertIn("t=1.0s", contents[1])
        self.assertIn("t=2.0s", contents[3])
        self.assertIn('"nose": [10.1, 20.5, 0.9]', contents[1])
        self.assertIn("200x120", contents[0])                 # image size, so coordinates have a scale

    def test_to_jpeg_downsizes_and_decodes(self):
        big = np.zeros((2000, 3000, 3), np.uint8)
        out = cv2.imdecode(np.frombuffer(gd.to_jpeg(big), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(max(out.shape[:2]), gd.MAX_IMAGE_SIDE)


class RunTests(unittest.TestCase):
    def run_one(self, generate=None, writer=None, pick=fake_pick, n=5):
        cap = {}
        row = gd.run("Gauri", "clip.mp4", n, pick=pick, pose_fn=fake_pose, generate=generate,
                     writer=writer, now=NOW, log=lambda m: None, capture=cap)
        return row, cap

    def test_scored_row_is_written(self):
        w = FakeWriter()
        row, _ = self.run_one(generate=lambda c: GOOD, writer=w)
        self.assertEqual(w.rows, [row])
        self.assertEqual(row["cow_name"], "Gauri")
        self.assertEqual(row["score"], 2)
        self.assertEqual(row["response"], "score=2 | slight head nod, short rear stride")
        self.assertEqual(row["recorded_at"], "2026-10-15T06:00:00+00:00")
        self.assertEqual(set(row), {n for n, _, _ in gd.BQ_SCHEMA})   # row matches the table schema

    def test_saved_image_is_the_best_ranked_frame_not_the_earliest(self):
        row, _ = self.run_one(generate=lambda c: GOOD)
        saved = cv2.imdecode(np.frombuffer(base64.b64decode(row["image"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertAlmostEqual(float(saved[2, 2].mean()), 10, delta=3)   # RANKED[0] has shade 10

    def test_gemini_sees_frames_in_time_order(self):
        _, cap = self.run_one(generate=lambda c: GOOD)
        times = [c for c in cap["contents"] if isinstance(c, str) and c.startswith("Frame")]
        self.assertEqual([t.split("t=")[1].split("s")[0] for t in times], ["0.5", "1.0", "2.0", "3.0", "4.0"])

    def test_untrusted_answer_writes_row_with_null_score(self):
        bad = json.dumps({"image_quality_ok": False, "gait_score": 4, "summary": "blurry"})
        w = FakeWriter()
        row, _ = self.run_one(generate=lambda c: bad, writer=w)
        self.assertIsNone(row["score"])
        self.assertTrue(row["response"].startswith("NO SCORE | image quality not ok"))
        self.assertEqual(len(w.rows), 1)

    def test_gemini_error_is_recorded_not_raised(self):
        def boom(c):
            raise RuntimeError("quota")

        w = FakeWriter()
        row, _ = self.run_one(generate=boom, writer=w)
        self.assertIsNone(row["score"])
        self.assertIn("ERROR RuntimeError: quota", row["response"])
        self.assertEqual(len(w.rows), 1)

    def test_dry_run_calls_nothing(self):
        row, cap = self.run_one(generate=None, writer=None)
        self.assertEqual(row["response"], "DRY RUN | Gemini not called")
        self.assertIsNone(row["score"])
        self.assertTrue(cap["contents"])

    def test_no_frames_returns_none_and_writes_nothing(self):
        w = FakeWriter()
        row, _ = self.run_one(generate=lambda c: GOOD, writer=w, pick=lambda v, n: [])
        self.assertIsNone(row)
        self.assertEqual(w.rows, [])

    def test_fewer_frames_than_requested_still_works(self):
        row, cap = self.run_one(generate=lambda c: GOOD, n=2)
        self.assertEqual(len(cap["frames"]), 2)


class FakeJob:
    def result(self):
        return None


class FakeBQ:
    def __init__(self):
        self.loads, self.datasets, self.tables = [], [], []

    def load_table_from_json(self, rows, table_id, job_config=None):
        self.loads.append((rows, table_id, job_config))
        return FakeJob()

    def create_dataset(self, ds, exists_ok=False):
        self.datasets.append(ds)

    def create_table(self, t, exists_ok=False):
        self.tables.append(t)


class BigQueryTests(unittest.TestCase):
    def test_add_is_a_json_serialisable_append_load_job(self):
        client = FakeBQ()
        w = gd.BigQueryWriter("proj", "ds", "tbl", client=client)
        row, _ = RunTests().run_one(generate=lambda c: GOOD)
        w.add(row)
        rows, table_id, cfg = client.loads[0]
        self.assertEqual(table_id, "proj.ds.tbl")
        self.assertEqual(cfg.write_disposition, "WRITE_APPEND")
        json.dumps(rows)
        self.assertEqual([f.name for f in cfg.schema], [n for n, _, _ in gd.BQ_SCHEMA])
        self.assertEqual({f.name: f.field_type for f in cfg.schema}["image"], "BYTES")

    def test_ensure_table_creates_dataset_and_table(self):
        client = FakeBQ()
        gd.BigQueryWriter("proj", "ds", "tbl", client=client).ensure_table()
        self.assertEqual(client.datasets[0].dataset_id, "ds")
        self.assertEqual(client.tables[0].table_id, "tbl")


class ConfigTests(unittest.TestCase):
    def test_placeholder_means_not_configured(self):
        self.assertFalse(gd.cloud_configured() and gd.PROJECT_ID == gd.PLACEHOLDER)


if __name__ == "__main__":
    unittest.main()
