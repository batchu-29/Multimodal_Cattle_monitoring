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

    def test_prompt_has_a_generic_indian_breed_note_without_naming_breeds(self):
        intro = gd.build_contents([RANKED[0]], fake_pose([0]))[0]
        self.assertIn("Indian cattle vary a lot in build", intro)
        self.assertIn("hump", intro)                        # natural anatomy, not lameness
        self.assertIn("Build alone must not raise the score", intro)
        self.assertIn("EARLY gait change", intro)           # the early-detection goal is still there
        for breed in ("Gir", "Sahiwal", "Tharparkar", "zebu", "Murrah"):
            self.assertNotIn(breed, intro)                  # generic: no hard-coded breeds

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

    def run_one_with_secret(self, generate, secret):
        row = gd.run("Gauri", "clip.mp4", 5, pick=fake_pick, pose_fn=fake_pose, generate=generate,
                     now=NOW, log=lambda m: None, secret=secret)
        return row, None

    def test_scored_row_is_written(self):
        w = FakeWriter()
        row, _ = self.run_one(generate=lambda c: GOOD, writer=w)
        self.assertEqual(w.rows, [row])
        self.assertEqual(row["cow_name"], "Gauri")
        self.assertEqual(row["score"], 2)
        self.assertEqual(row["response"], "score=2 | slight head nod, short rear stride")
        self.assertEqual(row["recorded_at"], NOW)
        self.assertEqual(set(row), set(gd.ROW_FIELDS))

    def test_saved_image_is_the_best_ranked_frame_not_the_earliest(self):
        row, _ = self.run_one(generate=lambda c: GOOD)
        self.assertIsInstance(row["image"], bytes)
        saved = cv2.imdecode(np.frombuffer(row["image"], np.uint8), cv2.IMREAD_COLOR)
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

    def test_dry_run_record_is_never_saved_even_with_a_writer(self):
        w = FakeWriter()
        row, _ = self.run_one(generate=None, writer=w)
        self.assertIsNotNone(row)
        self.assertEqual(w.rows, [])

    def test_api_key_is_redacted_from_error_records(self):
        def boom(c):
            raise RuntimeError("403 for https://x/y?key=SECRET123 denied")

        row, _ = self.run_one_with_secret(boom, "SECRET123")
        self.assertNotIn("SECRET123", row["response"])
        self.assertIn("***", row["response"])

    def test_no_frames_returns_none_and_writes_nothing(self):
        w = FakeWriter()
        row, _ = self.run_one(generate=lambda c: GOOD, writer=w, pick=lambda v, n: [])
        self.assertIsNone(row)
        self.assertEqual(w.rows, [])

    def test_fewer_frames_than_requested_still_works(self):
        row, cap = self.run_one(generate=lambda c: GOOD, n=2)
        self.assertEqual(len(cap["frames"]), 2)


class FakeDoc:
    def __init__(self, store, doc_id):
        self.store, self.doc_id = store, doc_id

    def set(self, data):
        self.store[self.doc_id] = data


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def document(self, doc_id):
        return FakeDoc(self.store, doc_id)


class FakeFirestore:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        return FakeCollection(self.collections.setdefault(name, {}))


class FirestoreTests(unittest.TestCase):
    def test_add_writes_one_document_keyed_by_record_id(self):
        client = FakeFirestore()
        w = gd.FirestoreWriter(client=client, collection="gait_records")
        row, _ = RunTests().run_one(generate=lambda c: GOOD)
        w.add(row)
        docs = client.collections["gait_records"]
        self.assertEqual(list(docs), [row["record_id"]])
        doc = docs[row["record_id"]]
        self.assertNotIn("record_id", doc)                 # the id is the document key, not a field
        self.assertEqual(set(doc), set(gd.ROW_FIELDS) - {"record_id"})
        self.assertEqual(doc["cow_name"], "Gauri")
        self.assertEqual(doc["score"], 2)
        self.assertIsInstance(doc["image"], bytes)
        self.assertEqual(doc["recorded_at"], NOW)          # a datetime: Firestore stores it as a Timestamp

    def test_null_score_is_kept_as_none(self):
        client = FakeFirestore()
        row, _ = RunTests().run_one(generate=lambda c: "garbage", writer=gd.FirestoreWriter(client=client))
        self.assertIsNone(client.collections["gait_records"][row["record_id"]]["score"])


def has_nested_array(obj, inside_list=False):
    """Firestore rejects arrays inside arrays; walk the record and look for one."""
    if isinstance(obj, (list, tuple)):
        return inside_list or any(has_nested_array(x, True) for x in obj)
    if isinstance(obj, dict):
        return any(has_nested_array(v, False) for v in obj.values())
    return False


class RetryTests(unittest.TestCase):
    def test_retries_temporary_errors_then_succeeds(self):
        calls, waits = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("503 UNAVAILABLE. The service is currently unavailable.")
            return "ok"

        self.assertEqual(gd.with_retries(flaky, delays=(1, 2, 3), sleep=waits.append), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(waits, [1, 2])

    def test_rate_limit_is_temporary(self):
        self.assertTrue(gd._is_temporary(RuntimeError("429 RESOURCE_EXHAUSTED")))

    def test_permanent_errors_are_not_retried(self):
        calls = []

        def broken():
            calls.append(1)
            raise RuntimeError("403 PERMISSION_DENIED")

        with self.assertRaises(RuntimeError):
            gd.with_retries(broken, delays=(1, 2), sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_the_last_delay_and_raises(self):
        calls = []

        def always_down():
            calls.append(1)
            raise RuntimeError("503 UNAVAILABLE")

        with self.assertRaises(RuntimeError):
            gd.with_retries(always_down, delays=(1, 2), sleep=lambda s: None)
        self.assertEqual(len(calls), 3)               # first try + 2 retries


class ImageInputTests(unittest.TestCase):
    def test_a_still_image_is_one_frame_at_time_zero(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "cow.PNG")           # extension check is case-insensitive
        cv2.imwrite(path, frame(77))
        frames = gd.pick_frames(path, 5)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][1], 0.0)
        self.assertEqual(frames[0][0].shape[:2], (120, 200))

    def test_unreadable_image_raises_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            gd.pick_frames(os.path.join(os.path.dirname(__file__), "no_such_photo.jpg"), 1)


class KeypointRecordTests(unittest.TestCase):
    def run_distinct(self):
        # a different x per frame, so the "best frame" keypoints can be told apart
        def pose_fn(frames):
            return [{"nose": (10.0 * (i + 1), 20.0, 0.9), "tail_base": (5.0, 5.0, 0.1)} for i in range(len(frames))]

        return gd.run("Gauri", "clip.mp4", 5, pick=fake_pick, pose_fn=pose_fn, generate=lambda c: GOOD,
                      now=NOW, log=lambda m: None)

    def test_keypoints_record_shape_keeps_low_confidence_points(self):
        kp = gd.keypoints_record({"nose": (10.123, 20.456, 0.9), "tail_base": (100, 50, 0.2)})
        self.assertEqual(kp["nose"], {"x": 10.1, "y": 20.5, "conf": 0.9})
        self.assertEqual(kp["tail_base"], {"x": 100, "y": 50, "conf": 0.2})

    def test_row_has_keypoints_for_best_frame_and_all_frames_in_time_order(self):
        row = self.run_distinct()
        self.assertEqual([f["time_s"] for f in row["frames"]], [0.5, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sorted(f["rank"] for f in row["frames"]), [1, 2, 3, 4, 5])
        best = next(f for f in row["frames"] if f["rank"] == 1)
        self.assertEqual(best["time_s"], 3.0)                       # RANKED[0] is the t=3.0s frame
        self.assertEqual(row["keypoints"], best["keypoints"])       # saved image <-> saved keypoints
        self.assertEqual(row["keypoints"]["nose"]["x"], 10.0)       # pose of ranked[0], not of the earliest

    def test_frame_size_is_stored_for_the_coordinates(self):
        row = self.run_distinct()
        self.assertEqual((row["frame_width"], row["frame_height"]), (200, 120))

    def test_record_has_no_nested_arrays_so_firestore_accepts_it(self):
        row = self.run_distinct()
        plain = {k: v for k, v in row.items() if not isinstance(v, bytes)}
        self.assertFalse(has_nested_array(plain))
        self.assertTrue(has_nested_array({"a": [[1, 2]]}))          # the checker itself works

    def test_whole_document_is_well_under_the_firestore_limit(self):
        row = self.run_distinct()
        size = len(row["image"]) + len(row["overlay_image"]) + len(json.dumps(
            {k: v for k, v in row.items() if not isinstance(v, (bytes, datetime))}))
        self.assertLess(size, 1_000_000)

    def test_firestore_document_carries_the_keypoints(self):
        client = FakeFirestore()
        row = RunTests().run_one(generate=lambda c: GOOD, writer=gd.FirestoreWriter(client=client))[0]
        doc = client.collections["gait_records"][row["record_id"]]
        self.assertEqual(doc["keypoints"], row["keypoints"])
        self.assertEqual(len(doc["frames"]), 5)


class OverlayTests(unittest.TestCase):
    base = np.full((120, 200, 3), 128, np.uint8)

    def test_does_not_modify_input_and_keeps_size(self):
        before = self.base.copy()
        out = gd.draw_overlay(self.base, {"nose": (50, 50, 0.9)})
        self.assertTrue(np.array_equal(self.base, before))
        self.assertEqual(out.shape, self.base.shape)
        self.assertFalse(np.array_equal(out, self.base))

    def test_empty_pose_leaves_image_unchanged(self):
        self.assertTrue(np.array_equal(gd.draw_overlay(self.base, {}), self.base))

    def test_confident_point_is_green_and_low_confidence_is_red(self):
        out = gd.draw_overlay(self.base, {"nose": (50, 50, 0.9), "tail_base": (150, 80, 0.1)})
        self.assertEqual(tuple(out[50, 50]), (0, 200, 0))     # BGR green
        self.assertEqual(tuple(out[80, 150]), (0, 0, 255))    # BGR red

    def test_skeleton_line_joins_two_confident_points_only(self):
        both = gd.draw_overlay(self.base, {"nose": (20, 60, 0.9), "neck_base": (180, 60, 0.9)})
        # the line is anti-aliased, so check it is close to yellow rather than exactly yellow
        self.assertLess(np.abs(both[60, 100].astype(int) - np.array([0, 220, 255])).max(), 40)
        one_weak = gd.draw_overlay(self.base, {"nose": (20, 60, 0.9), "neck_base": (180, 60, 0.1)})
        self.assertEqual(tuple(one_weak[60, 100]), (128, 128, 128))  # no line to a weak point

    def test_run_stores_a_decodable_overlay_that_differs_from_the_image(self):
        row, _ = RunTests().run_one(generate=lambda c: GOOD)
        img = cv2.imdecode(np.frombuffer(row["image"], np.uint8), cv2.IMREAD_COLOR)
        ov = cv2.imdecode(np.frombuffer(row["overlay_image"], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, ov.shape)
        self.assertGreater(float(np.abs(img.astype(int) - ov.astype(int)).sum()), 0)

    def test_both_images_fit_in_one_firestore_document(self):
        noisy = np.random.RandomState(1).randint(0, 255, (1500, 2000, 3), dtype=np.uint8)
        total = len(gd.fit_jpeg(noisy)) + len(gd.fit_jpeg(gd.draw_overlay(noisy, {"nose": (9, 9, 1.0)})))
        self.assertLess(total, 1_000_000)     # the document limit is 1 MiB (1,048,576)


class ImageSizeTests(unittest.TestCase):
    def test_fit_jpeg_stays_under_the_firestore_limit(self):
        noisy = np.random.RandomState(0).randint(0, 255, (1500, 2000, 3), dtype=np.uint8)
        data = gd.fit_jpeg(noisy)
        self.assertLessEqual(len(data), gd.MAX_STORED_IMAGE_BYTES)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = (gd.GEMINI_API_KEY, gd.FIREBASE_CREDENTIALS, gd.GCP_PROJECT_ID, gd.GEMINI_BACKEND,
                       gd.adc_available)
        gd.GEMINI_API_KEY, gd.GCP_PROJECT_ID, gd.GEMINI_BACKEND = "", "", ""   # ignore the real .env

    def tearDown(self):
        (gd.GEMINI_API_KEY, gd.FIREBASE_CREDENTIALS, gd.GCP_PROJECT_ID, gd.GEMINI_BACKEND,
         gd.adc_available) = self._saved

    def test_gemini_configured_needs_a_real_key(self):
        for value, expected in (("", False), ("YOUR_GEMINI_API_KEY", False), ("AIza-real-looking", True)):
            gd.GEMINI_API_KEY = value
            self.assertEqual(gd.gemini_configured(), expected, value)

    def test_backend_selection(self):
        cases = [  # (api key, project, forced backend) -> backend
            (("", "", ""), None),
            (("KEY", "", ""), "api_key"),
            (("", "cattle-id", ""), "vertex"),
            (("KEY", "cattle-id", ""), "vertex"),                 # a project wins in auto mode
            (("KEY", "cattle-id", "api_key"), "api_key"),         # unless forced
            (("KEY", "", "vertex"), None),                        # forced vertex but no project
            (("", "YOUR_GCP_PROJECT", ""), None),                 # untouched placeholder
        ]
        for (key, project, forced), expected in cases:
            gd.GEMINI_API_KEY, gd.GCP_PROJECT_ID, gd.GEMINI_BACKEND = key, project, forced
            self.assertEqual(gd.gemini_backend(), expected, (key, project, forced))

    def test_vertex_needs_a_gcloud_login_to_count_as_configured(self):
        gd.GCP_PROJECT_ID = "cattle-id"
        gd.adc_available = lambda: False
        self.assertFalse(gd.gemini_configured())
        self.assertIn("gcloud auth application-default login", gd.gemini_status())
        gd.adc_available = lambda: True
        self.assertTrue(gd.gemini_configured())
        self.assertIn("Vertex AI, project cattle-id", gd.gemini_status())

    def test_status_when_nothing_is_set_names_both_options(self):
        status = gd.gemini_status()
        self.assertIn("GCP_PROJECT_ID", status)
        self.assertIn("GEMINI_API_KEY", status)

    def test_vertex_client_is_built_with_project_and_location(self):
        from unittest import mock

        gd.GCP_PROJECT_ID = "cattle-id"
        with mock.patch("google.genai.Client") as client:
            gd.gemini_generate()
        client.assert_called_once_with(vertexai=True, project="cattle-id", location=gd.GCP_LOCATION)

    def test_api_key_client_is_built_with_the_key_only(self):
        from unittest import mock

        gd.GEMINI_API_KEY = "KEY"
        with mock.patch("google.genai.Client") as client:
            gd.gemini_generate()
        client.assert_called_once_with(api_key="KEY")

    def test_firebase_configured_needs_an_existing_file(self):
        import tempfile

        gd.FIREBASE_CREDENTIALS = ""
        self.assertFalse(gd.firebase_configured())
        gd.FIREBASE_CREDENTIALS = "YOUR_PATH_TO_FIREBASE_SERVICE_ACCOUNT_JSON"
        self.assertFalse(gd.firebase_configured())
        gd.FIREBASE_CREDENTIALS = os.path.join(tempfile.gettempdir(), "does_not_exist_12345.json")
        self.assertFalse(gd.firebase_configured())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            fh.write(b"{}")
        try:
            gd.FIREBASE_CREDENTIALS = fh.name
            self.assertTrue(gd.firebase_configured())
        finally:
            os.unlink(fh.name)

    def test_env_file_with_a_bom_still_reads_the_first_variable(self):
        import tempfile

        with tempfile.NamedTemporaryFile("wb", suffix=".env", delete=False) as fh:
            fh.write(b"\xef\xbb\xbfTEST_GAIT_BOM=first_line_value\nTEST_GAIT_BOM2=second\n")   # UTF-8 with BOM
        try:
            os.environ.pop("TEST_GAIT_BOM", None)
            gd.load_env_file(fh.name)
            self.assertEqual(os.environ.get("TEST_GAIT_BOM"), "first_line_value")
            self.assertNotIn("﻿TEST_GAIT_BOM", os.environ)
        finally:
            os.unlink(fh.name)
            for k in ("TEST_GAIT_BOM", "TEST_GAIT_BOM2"):
                os.environ.pop(k, None)

    def test_env_file_loader_sets_missing_but_never_overrides(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
            fh.write("# comment\nTEST_GAIT_A=from_file\nTEST_GAIT_B='quoted'\nTEST_GAIT_C=from_file\n\nbad line\n")
        try:
            os.environ.pop("TEST_GAIT_A", None)
            os.environ.pop("TEST_GAIT_B", None)
            os.environ["TEST_GAIT_C"] = "already_set"
            gd.load_env_file(fh.name)
            self.assertEqual(os.environ["TEST_GAIT_A"], "from_file")
            self.assertEqual(os.environ["TEST_GAIT_B"], "quoted")
            self.assertEqual(os.environ["TEST_GAIT_C"], "already_set")
            gd.load_env_file(fh.name + ".missing")     # a missing file is fine
        finally:
            os.unlink(fh.name)
            for k in ("TEST_GAIT_A", "TEST_GAIT_B", "TEST_GAIT_C"):
                os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
