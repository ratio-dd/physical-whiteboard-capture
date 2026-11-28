import time
import unittest

import numpy as np

from live_ink.runtime import OrderedInkRuntime, _official_force_binary


class _Backend:
    def predict(self, bgr):
        return (bgr[:, :, 0] > 0).astype(np.float32)


class _Pipeline:
    def __init__(self): self.seqs = []
    def step(self, bgr, known, score):
        self.seqs.append(int(bgr[0, 0, 0]))
        return {"seq": int(bgr[0, 0, 0]), "pixels": int(score.sum())}


class RuntimeTests(unittest.TestCase):
    def test_official_force_binary_uses_uint8_128_boundary(self):
        values = np.array([[127 / 255, 128 / 255, 129 / 255]], np.float32)
        np.testing.assert_array_equal(_official_force_binary(values), [[1, 0, 0]])

    def setUp(self):
        self.pipelines = []
        def factory():
            p = _Pipeline(); self.pipelines.append(p); return p
        self.runtime = OrderedInkRuntime(_Backend(), factory,
                                         lambda result, w, h: f"<svg data-seq='{result['seq']}'/>",
                                         max_queue=3)
        self.runtime.start("test-session")

    def tearDown(self):
        self.runtime.stop()

    def frame(self, seq):
        bgr = np.zeros((4, 5, 3), np.uint8); bgr[:, :, 0] = seq
        return bgr, np.full((4, 5), 255, np.uint8)

    def test_order_and_result(self):
        for seq in (1, 2, 3):
            bgr, known = self.frame(seq); self.runtime.submit(seq, seq, bgr, known)
        out = [self.runtime.get_result(2) for _ in range(3)]
        self.assertEqual([r.seq for r in out], [1, 2, 3])
        self.assertEqual([r.pipeline_result["pixels"] for r in out], [20, 20, 20])
        self.assertEqual(self.runtime.stats().dropped, 0)

    def test_rejects_regressing_sequence(self):
        bgr, known = self.frame(1); self.runtime.submit(2, 2, bgr, known)
        with self.assertRaises(ValueError): self.runtime.submit(1, 1, bgr, known)

    def test_restart_resets_session_identity(self):
        bgr, known = self.frame(1); self.runtime.submit(1, 1, bgr, known)
        self.assertEqual(self.runtime.get_result(2).session_id, "test-session")
        self.runtime.stop(); self.runtime.start("second-session")
        bgr, known = self.frame(1); self.runtime.submit(1, 1, bgr, known)
        self.assertEqual(self.runtime.get_result(2).session_id, "second-session")

    def test_restart_discards_unread_old_results(self):
        bgr, known = self.frame(1); self.runtime.submit(1, 1, bgr, known)
        deadline = time.time() + 2
        while self.runtime.stats().completed < 1 and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.runtime.stats().completed, 1)
        self.runtime.stop(); self.runtime.start("new-session")
        bgr, known = self.frame(2); self.runtime.submit(1, 2, bgr, known)
        result = self.runtime.get_result(2)
        self.assertEqual(result.session_id, "new-session")
        self.assertEqual(result.seq, 1)

    def test_unread_results_are_bounded_and_stop_drains(self):
        for seq in range(1, 12):
            bgr, known = self.frame(seq)
            self.runtime.submit(seq, seq, bgr, known)
        self.runtime.stop(timeout=2)
        self.assertLessEqual(self.runtime._results.qsize(), self.runtime.max_queue)
        self.assertEqual(self.runtime.stats().state, "stopped")

    def test_official_binary_polarity_is_black_ink_positive(self):
        probability = np.array([[0.0, 0.49, 0.51, 1.0]], np.float32)
        np.testing.assert_array_equal(_official_force_binary(probability), [[1.0, 1.0, 0.0, 0.0]])


if __name__ == "__main__": unittest.main()
