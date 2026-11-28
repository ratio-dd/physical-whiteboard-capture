import unittest
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from weak_ink_model import load_model

_SPEC = importlib.util.spec_from_file_location(
    "lecturemath_vector_pipeline",
    Path(__file__).parents[1] / "experiments" / "lecturemath-ink" / "vector_pipeline.py",
)
_PIPELINE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PIPELINE
_SPEC.loader.exec_module(_PIPELINE)
candidate_mask = _PIPELINE.candidate_mask
extract_hybrid = _PIPELINE.extract_hybrid


class VectorPipelineTests(unittest.TestCase):
    class MockModel:
        model_version = "mock"
        def __init__(self, mask): self.mask = mask
        def predict_proba(self, bgr, known=None): return self.mask.astype(np.float32)

    def test_candidates_are_known_source_supported(self):
        board = np.full((80, 120, 3), 245, np.uint8)
        cv2.line(board, (15, 40), (105, 40), (220, 220, 220), 2)
        known = np.full((80, 120), 255, np.uint8)
        known[:, :10] = 0
        model = load_model("models/weak_ink_v2.npz")
        mask = candidate_mask(model, board, known)
        self.assertEqual(mask[:, :10].max(), 0)
        self.assertEqual(mask.dtype, np.uint8)

    def test_hybrid_returns_geometry_not_mask(self):
        board = np.full((100, 140, 3), 245, np.uint8)
        cv2.line(board, (15, 45), (120, 45), (150, 150, 150), 2)
        known = np.full((100, 140), 255, np.uint8)
        result = extract_hybrid(load_model("models/weak_ink_v2.npz"), board, known)
        self.assertEqual(result.model_version, "weak_ink_v2")
        self.assertEqual(result.candidate_mask.dtype, np.uint8)
        self.assertIsInstance(result.vector_paths, list)
        self.assertTrue(all("points" in path for path in result.vector_paths))

    def test_unknown_storage_cannot_become_candidate(self):
        board = np.full((40, 40, 3), 0, np.uint8)
        known = np.zeros((40, 40), np.uint8)
        model = load_model("models/weak_ink_v2.npz")
        self.assertEqual(int(candidate_mask(model, board, known).max()), 0)

    def test_explicit_model_mask_changes_export_geometry(self):
        board = np.full((80, 120, 3), 245, np.uint8)
        cv2.line(board, (20, 40), (100, 40), (180, 180, 180), 2)
        known = np.full((80, 120), 255, np.uint8)
        empty = self.MockModel(np.zeros((80, 120), np.float32))
        zero = extract_hybrid(empty, board, known, threshold=.5)
        classic = _PIPELINE.extract_paths(board, known, mode="classic")
        self.assertEqual(zero.vector_paths, classic)

        mask_a = np.zeros((80, 120), np.float32); mask_a[35:42, 25:40] = 1
        mask_b = np.zeros((80, 120), np.float32); mask_b[35:42, 70:85] = 1
        a = extract_hybrid(self.MockModel(mask_a), board, known, threshold=.5)
        b = extract_hybrid(self.MockModel(mask_b), board, known, threshold=.5)
        self.assertGreater(a.added_paths, 0)
        self.assertGreater(b.added_paths, 0)
        self.assertNotEqual(a.vector_paths[-1]["points"], b.vector_paths[-1]["points"])

    def test_same_line_is_deduplicated_but_new_segments_survive(self):
        board = np.full((100, 160, 3), 245, np.uint8)
        cv2.line(board, (15, 25), (70, 25), (130, 130, 130), 3)
        # A rim-touching source line is intentionally rejected by classic's
        # border-rail filter, while remaining valid model-supported evidence.
        cv2.line(board, (5, 1), (60, 1), (180, 180, 180), 2)
        known = np.full((100, 160), 255, np.uint8)
        same = np.zeros((100, 160), np.float32); same[23:28, 15:71] = 1
        result = extract_hybrid(self.MockModel(same), board, known, threshold=.5)
        self.assertEqual(result.added_paths, 0)
        new = np.zeros((100, 160), np.float32); new[0:4, 5:61] = 1
        result = extract_hybrid(self.MockModel(new), board, known, threshold=.5)
        self.assertGreater(result.added_paths, 0)

    def test_ring_hole_remains_transparent_after_render(self):
        board = np.full((100, 100, 3), 245, np.uint8)
        cv2.circle(board, (50, 50), 22, (180, 180, 180), 3)
        known = np.full((100, 100), 255, np.uint8)
        mask = np.zeros((100, 100), np.float32)
        cv2.circle(mask, (50, 50), 22, 1, 3)
        result = extract_hybrid(self.MockModel(mask), board, known, threshold=.5)
        candidate = result.vector_paths[-1]
        self.assertGreaterEqual(len(candidate["outline"]), 2)
        rendered = np.zeros((100, 100), np.uint8)
        # Explicit parity render for the hole check.
        rendered[:] = 0
        for ring in candidate["outline"]:
            layer = np.zeros_like(rendered)
            cv2.fillPoly(layer, [np.rint(np.asarray(ring)).astype(np.int32)], 255)
            rendered = cv2.bitwise_xor(rendered, layer)
        self.assertEqual(int(rendered[50, 50]), 0)

    def test_independent_and_crossing_components_remain_editable(self):
        board = np.full((100, 120, 3), 245, np.uint8)
        cv2.line(board, (5, 1), (35, 1), (180, 180, 180), 2)
        cv2.line(board, (80, 1), (105, 1), (180, 180, 180), 2)
        cv2.line(board, (92, 0), (92, 12), (180, 180, 180), 2)
        known = np.full((100, 120), 255, np.uint8)
        mask = np.zeros((100, 120), np.float32)
        mask[0:4, 5:36] = 1; mask[0:4, 80:106] = 1; mask[0:13, 90:95] = 1
        result = extract_hybrid(self.MockModel(mask), board, known, threshold=.5)
        self.assertGreaterEqual(result.added_paths, 2)
        self.assertTrue(all(path["kind"] == "model_candidate" for path in result.vector_paths[len(result.baseline_paths):]))


if __name__ == "__main__":
    unittest.main()
