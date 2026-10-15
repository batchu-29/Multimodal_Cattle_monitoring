import json
import math
import unittest
from datetime import datetime, timezone

from cattle_monitor.daily import run_daily
from cattle_monitor.gait import compute_gait
from cattle_monitor.gait_score import build_prompt, parse_assessment
from cattle_monitor.screening import Screening, screen
from cattle_monitor.store import BigQueryStore, InMemoryStore, GaitRecord

FPS = 30
N = 150  # 5 s


def synth_poses(head_amp=0.02, left_amp=0.15, right_amp=0.15, arch=0.0, direction=1, conf=0.9, n=N):
    """Walking cow, trunk length 100 px. Amplitudes are fractions of trunk length."""
    poses = []
    for i in range(n):
        t = i / FPS
        cx = 100 + direction * 60 * t * 1.0           # moves 60 px/s along x
        cy = 300.0
        w = 2 * math.pi * 1.0 * t                      # 1 Hz stride
        d = direction

        def P(dx, dy):
            return (cx + d * dx, cy + dy, conf)

        pose = {
            "front_left_thigh": P(50, 0), "front_right_thigh": P(50, 0),
            "hind_left_thigh": P(-50, 0), "hind_right_thigh": P(-50, 0),
            "nose": P(110, -60 + head_amp * 100 * math.sin(2 * w)),
            "neck_base": P(40, -50),
            "tail_base": P(-60, -50),
            "back_middle": P(-10, -50 - arch * 100),
            "hind_left_paw": P(-50 + left_amp * 50 * math.sin(w), 80),
            "hind_right_paw": P(-50 + right_amp * 50 * math.sin(w + math.pi), 80),
        }
        poses.append(pose)
    return poses


class GaitTests(unittest.TestCase):
    def test_symmetric_flat_back_is_low(self):
        m = compute_gait(synth_poses(), FPS)
        self.assertTrue(m.reliable)
        self.assertLess(m.hind_asymmetry, 0.05)
        self.assertLess(abs(m.back_arch), 0.01)

    def test_head_bob_scales_with_amplitude(self):
        small = compute_gait(synth_poses(head_amp=0.02), FPS).head_bob
        big = compute_gait(synth_poses(head_amp=0.10), FPS).head_bob
        self.assertGreater(big, 3 * small)

    def test_asymmetry_detected(self):
        m = compute_gait(synth_poses(left_amp=0.30, right_amp=0.10), FPS)
        self.assertAlmostEqual(m.hind_asymmetry, 1.0, delta=0.1)  # |0.3-0.1| / 0.2

    def test_back_arch_positive_when_arched(self):
        m = compute_gait(synth_poses(arch=0.30), FPS)
        self.assertAlmostEqual(m.back_arch, 0.30, delta=0.01)

    def test_direction_independent(self):
        a = compute_gait(synth_poses(direction=1, left_amp=0.3, right_amp=0.1), FPS)
        b = compute_gait(synth_poses(direction=-1, left_amp=0.3, right_amp=0.1), FPS)
        self.assertAlmostEqual(a.hind_asymmetry, b.hind_asymmetry, places=2)
        self.assertAlmostEqual(a.head_bob, b.head_bob, places=2)

    def test_too_few_frames_unreliable(self):
        m = compute_gait(synth_poses(n=10), FPS)
        self.assertFalse(m.reliable)

    def test_low_confidence_unreliable(self):
        m = compute_gait(synth_poses(conf=0.1), FPS)
        self.assertFalse(m.reliable)
        self.assertEqual(m.n_valid, 0)

    def test_not_walking_unreliable(self):
        poses = synth_poses()
        # keep x constant per keypoint (cow standing in place)
        base = poses[0]
        still = [{k: (base[k][0], v[1], v[2]) for k, v in p.items()} for p in poses]
        m = compute_gait(still, FPS)
        self.assertFalse(m.reliable)
        self.assertIn("not walking", m.reason)


class ScreenTests(unittest.TestCase):
    def test_sound_cow_not_flagged(self):
        self.assertFalse(screen(compute_gait(synth_poses(), FPS)).flagged)

    def test_any_single_metric_flags(self):
        self.assertTrue(screen(compute_gait(synth_poses(left_amp=0.3, right_amp=0.1), FPS)).flagged)
        self.assertTrue(screen(compute_gait(synth_poses(arch=0.30), FPS)).flagged)
        self.assertTrue(screen(compute_gait(synth_poses(head_amp=0.2), FPS)).flagged)


GOOD = json.dumps({"image_quality_ok": True, "gait_score": 3, "confidence": 0.7,
                   "observations": ["arched back"], "rationale": "short hind strides"})


class GateTests(unittest.TestCase):
    def test_good_response_parsed(self):
        a = parse_assessment(GOOD)
        self.assertEqual(a.score, 3)
        self.assertEqual(a.observations, ["arched back"])

    def test_rejections(self):
        def resp(**kw):
            d = json.loads(GOOD)
            d.update(kw)
            return json.dumps(d)

        for bad in (
            resp(image_quality_ok=False),
            resp(image_quality_ok="true"),
            resp(gait_score=0),
            resp(gait_score=6),
            resp(gait_score=3.5),
            resp(gait_score=True),
            resp(gait_score="3"),
            "not json",
            "[]",
            None,
        ):
            self.assertIsNone(parse_assessment(bad), bad)

    def test_prompt_contains_numbers(self):
        m = compute_gait(synth_poses(left_amp=0.3, right_amp=0.1), FPS)
        prompt = build_prompt(m, screen(m))
        self.assertIn("hind_asymmetry", prompt)
        self.assertIn("Sprecher 1997", prompt)
        self.assertIn("EARLY gait change", prompt)


class FakeFrame:
    def __call__(self, video):
        return b"jpeg-bytes", 2.5


class PipelineTests(unittest.TestCase):
    NOW = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)

    def run_one(self, poses, generate=None, pick=None, store=None):
        store = store or InMemoryStore()
        rec = run_daily("Gauri", "clip.mp4", lambda v: (FPS, poses), store,
                        pick_frame=pick or FakeFrame(), generate=generate, now=self.NOW)
        return rec, store

    def test_normal_cow_no_gemini_call(self):
        calls = []
        rec, store = self.run_one(synth_poses(), generate=lambda j, p: calls.append(1) or GOOD)
        self.assertEqual(rec.status, "rules_normal")
        self.assertIsNone(rec.gait_score)
        self.assertEqual(calls, [])
        self.assertEqual(len(store.rows), 1)

    def test_flagged_cow_scored_by_gemini(self):
        seen = {}

        def gen(jpeg, prompt):
            seen["jpeg"], seen["prompt"] = jpeg, prompt
            return GOOD

        rec, _ = self.run_one(synth_poses(arch=0.30), generate=gen)
        self.assertEqual(rec.status, "scored")
        self.assertEqual(rec.gait_score, 3)
        self.assertTrue(rec.gemini_called)
        self.assertEqual(rec.frame_time_s, 2.5)
        self.assertEqual(seen["jpeg"], b"jpeg-bytes")
        self.assertIn("back_arch", rec.rule_reasons)

    def test_untrusted_gemini_answer_stores_no_score(self):
        bad = json.dumps({"image_quality_ok": False, "gait_score": 4, "confidence": 1,
                          "observations": [], "rationale": "blurry"})
        rec, _ = self.run_one(synth_poses(arch=0.30), generate=lambda j, p: bad)
        self.assertEqual(rec.status, "gemini_rejected")
        self.assertIsNone(rec.gait_score)

    def test_gemini_exception_recorded_not_raised(self):
        def boom(j, p):
            raise RuntimeError("quota")

        rec, store = self.run_one(synth_poses(arch=0.30), generate=boom)
        self.assertEqual(rec.status, "gemini_error")
        self.assertIn("quota", rec.gemini_rationale)
        self.assertEqual(len(store.rows), 1)

    def test_flagged_without_gemini_configured(self):
        rec, _ = self.run_one(synth_poses(arch=0.30), generate=None)
        self.assertEqual(rec.status, "gemini_skipped")

    def test_no_clean_frame(self):
        rec, _ = self.run_one(synth_poses(arch=0.30), generate=lambda j, p: GOOD, pick=lambda v: None)
        self.assertEqual(rec.status, "flagged_no_frame")
        self.assertFalse(rec.gemini_called)

    def test_bad_pose_is_insufficient_data_not_normal(self):
        rec, _ = self.run_one(synth_poses(conf=0.1))
        self.assertEqual(rec.status, "insufficient_data")
        self.assertFalse(rec.rule_flag)

    def test_history_newest_first(self):
        store = InMemoryStore()
        for day in (1, 2, 3):
            run_daily("Gauri", "c.mp4", lambda v: (FPS, synth_poses()), store, pick_frame=FakeFrame(),
                      now=datetime(2026, 10, day, tzinfo=timezone.utc))
        run_daily("Other", "c.mp4", lambda v: (FPS, synth_poses()), store, pick_frame=FakeFrame(), now=self.NOW)
        hist = store.history("Gauri")
        self.assertEqual(len(hist), 3)
        self.assertTrue(hist[0]["recorded_at"] > hist[1]["recorded_at"])


def hist(values, status="rules_normal", metric="back_arch", score=None):
    return [{"status": status, metric: v, "gait_score": score} for v in values]


class BaselineTests(unittest.TestCase):
    def metrics(self, arch):
        from cattle_monitor.gait import GaitMetrics
        return GaitMetrics(3, 3, True, True, "", None, None, arch)

    STEADY = [0.14, 0.15, 0.16, 0.15, 0.14, 0.16, 0.15]   # sound zebu with a hump: median 0.15

    def test_hump_is_normal_for_her_even_though_it_is_high(self):
        # 0.16 is within her normal range; no flag
        s = screen(self.metrics(0.16), history=hist(self.STEADY))
        self.assertFalse(s.flagged)

    def test_rise_over_own_baseline_flags_below_absolute_line(self):
        # 0.20 is not > the 0.20 absolute line, but it is well above her 0.15 baseline
        s = screen(self.metrics(0.20), history=hist(self.STEADY))
        self.assertTrue(s.flagged)
        self.assertIn("above her baseline", s.reasons[0])

    def test_cold_start_uses_absolute_line_only(self):
        self.assertFalse(screen(self.metrics(0.15), history=hist([0.15] * 4)).flagged)   # 4 < min_history
        s = screen(self.metrics(0.25), history=hist([0.15] * 4))
        self.assertTrue(s.flagged)
        self.assertIn("no baseline yet", s.reasons[0])

    def test_sound_looking_values_from_real_clips_do_not_flag_cold(self):
        self.assertFalse(screen(self.metrics(0.121)).flagged)
        self.assertFalse(screen(self.metrics(0.153)).flagged)

    def test_lame_days_do_not_poison_baseline(self):
        from cattle_monitor.screening import baselines_from_history
        rows = hist(self.STEADY) + hist([0.35] * 10, status="scored", score=4) + hist([0.40] * 10, status="gemini_skipped")
        b = baselines_from_history(rows)["back_arch"]
        self.assertAlmostEqual(b.median, 0.15, places=3)
        self.assertEqual(b.n, 7)

    def test_scored_1_or_2_counts_as_normal(self):
        from cattle_monitor.screening import baselines_from_history
        rows = hist([0.15] * 3) + hist([0.15] * 2, status="scored", score=2)
        self.assertEqual(baselines_from_history(rows)["back_arch"].n, 5)

    def test_pipeline_uses_store_history(self):
        import numpy as np
        from cattle_monitor.daily import run_daily_frames

        store = InMemoryStore()
        for day, v in enumerate(self.STEADY, 1):
            store.add(GaitRecord(f"r{day}", "Gauri", datetime(2026, 10, day, tzinfo=timezone.utc),
                                     "rules_normal", "c.mp4", 3, False, back_arch=v))
        img = np.zeros((64, 64, 3), np.uint8)
        # 0.20 arch: only her own baseline can flag it (absolute line is 0.20)
        rec = run_daily_frames("Gauri", "c.mp4", lambda v: [(img, 1.0), (img, 2.0)],
                               lambda f: synth_poses(arch=0.20, n=len(f)), store,
                               generate=lambda j, p: GOOD, now=datetime(2026, 10, 9, tzinfo=timezone.utc))
        self.assertEqual(rec.status, "scored")
        self.assertIn("above her baseline", rec.rule_reasons)

    def test_store_failure_falls_back_to_absolute_lines(self):
        from cattle_monitor.daily import _history

        class Broken:
            def history(self, *a, **k):
                raise RuntimeError("table not found")

        self.assertEqual(_history(Broken(), "Gauri"), [])


class FewFrameTests(unittest.TestCase):
    NOW = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)

    def run_one(self, poses, generate=None, picked=None):
        import numpy as np
        from cattle_monitor.daily import run_daily_frames

        img = np.zeros((64, 64, 3), np.uint8)
        picked = [(img, 1.0), (img, 2.0), (img, 3.0)] if picked is None else picked
        store = InMemoryStore()
        seen = []
        rec = run_daily_frames("Gauri", "clip.mp4", lambda v: picked,
                               lambda frames: seen.append(len(frames)) or poses[:len(frames)],
                               store, generate=generate, now=self.NOW)
        return rec, store, seen

    def test_static_only_measures_back_arch(self):
        from cattle_monitor.gait import compute_static
        m = compute_static(synth_poses(arch=0.30, n=3))
        self.assertTrue(m.reliable)
        self.assertAlmostEqual(m.back_arch, 0.30, delta=0.01)
        self.assertIsNone(m.head_bob)
        self.assertIsNone(m.hind_asymmetry)

    def test_static_needs_two_good_frames(self):
        from cattle_monitor.gait import compute_static
        self.assertFalse(compute_static(synth_poses(n=1)).reliable)
        self.assertFalse(compute_static(synth_poses(conf=0.1, n=3)).reliable)

    def test_flat_back_normal_no_gemini(self):
        calls = []
        rec, store, seen = self.run_one(synth_poses(n=3), generate=lambda j, p: calls.append(1) or GOOD)
        self.assertEqual(rec.status, "rules_normal")
        self.assertEqual(seen, [3])
        self.assertEqual(calls, [])

    def test_arched_back_goes_to_gemini_with_best_frame(self):
        prompts = []
        rec, _, _ = self.run_one(synth_poses(arch=0.30, n=3),
                                 generate=lambda j, p: prompts.append(p) or GOOD)
        self.assertEqual(rec.status, "scored")
        self.assertEqual(rec.frame_time_s, 1.0)          # first = best-ranked
        self.assertIn("not measured", prompts[0])        # head bob / asymmetry honestly absent

    def test_no_frames_is_insufficient(self):
        rec, store, _ = self.run_one([], picked=[])
        self.assertEqual(rec.status, "insufficient_data")
        self.assertEqual(len(store.rows), 1)


class SpreadTests(unittest.TestCase):
    def test_min_gap_skips_near_duplicates(self):
        from cattle_monitor.frame_select import Candidate, Detection, _spread
        box = Detection(0, 0, 1, 1, 1)
        cands = [Candidate(i, t, box, 1, 1, 1, score=s) for i, (t, s) in
                 enumerate([(1.0, .9), (1.1, .89), (1.2, .88), (2.0, .8), (3.0, .7)])]
        picked = _spread(cands, 3, 0.5)
        self.assertEqual([c.time_s for c in picked], [1.0, 2.0, 3.0])


class FakeJob:
    def result(self):
        return None


class FakeBQClient:
    def __init__(self):
        self.loads = []

    def load_table_from_json(self, rows, table_id, job_config=None):
        self.loads.append((rows, table_id, job_config))
        return FakeJob()


class BigQueryStoreTests(unittest.TestCase):
    def test_add_uses_load_job_with_serialisable_row(self):
        client = FakeBQClient()
        store = BigQueryStore("proj", "ds", "tbl", client=client)
        rec = GaitRecord("id1", "Gauri", datetime(2026, 10, 15, tzinfo=timezone.utc),
                             "scored", "c.mp4", 120, True, "back_arch 0.08 > 0.04", gait_score=3)
        store.add(rec)
        rows, table_id, cfg = client.loads[0]
        self.assertEqual(table_id, "proj.ds.tbl")
        self.assertEqual(rows[0]["cow_name"], "Gauri")
        self.assertEqual(rows[0]["recorded_at"], "2026-10-15T00:00:00+00:00")
        json.dumps(rows)  # must be JSON-serialisable
        self.assertEqual(cfg.write_disposition, "WRITE_APPEND")


if __name__ == "__main__":
    unittest.main()
