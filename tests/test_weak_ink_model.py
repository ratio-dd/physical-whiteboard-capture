import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from weak_ink_model import FEATURES, V2_FEATURES, WeakInkModel, WeakInkV2Model, candidate_mask, _features, _features_v2


class WeakInkModelTests(unittest.TestCase):
    def test_features_and_probability_shape(self):
        image = np.full((24, 32, 3), 240, np.uint8)
        weights = {"w1": np.zeros((FEATURES, 4), np.float32), "b1": np.zeros(4, np.float32),
                   "w2": np.zeros((4, 1), np.float32), "b2": np.zeros(1, np.float32)}
        model = WeakInkModel(weights)
        result = model.predict_proba(image)
        self.assertEqual(result.shape, image.shape[:2])
        self.assertTrue(np.allclose(result, .5))

    def test_candidate_mask_validates_threshold(self):
        image = np.full((8, 8, 3), 240, np.uint8)
        weights = {"w1": np.zeros((FEATURES, 2), np.float32), "b1": np.zeros(2, np.float32),
                   "w2": np.zeros((2, 1), np.float32), "b2": np.zeros(1, np.float32)}
        with self.assertRaises(ValueError): candidate_mask(WeakInkModel(weights), image, 1.0)

    def test_invalid_image(self):
        with self.assertRaises(ValueError): _features(np.zeros((8, 8), np.uint8))

    def test_unknown_pixels_are_never_candidates(self):
        image = np.full((8, 8, 3), 240, np.uint8)
        weights = {"w1": np.zeros((FEATURES, 2), np.float32), "b1": np.zeros(2, np.float32),
                   "w2": np.zeros((2, 1), np.float32), "b2": np.ones(1, np.float32) * 4}
        model = WeakInkModel(weights)
        known = np.zeros((8, 8), np.uint8)
        known[:, :4] = 255
        result = model.predict_proba(image, known)
        self.assertTrue(np.all(result[:, 4:] == 0))
        self.assertTrue(np.all(result[:, :4] > 0.5))

    def test_one_pixel_candidate_is_preserved(self):
        image = np.full((8, 8, 3), 240, np.uint8)
        weights = {"w1": np.zeros((FEATURES, 2), np.float32), "b1": np.zeros(2, np.float32),
                   "w2": np.zeros((2, 1), np.float32), "b2": np.ones(1, np.float32) * 4}
        mask = candidate_mask(WeakInkModel(weights), image, .7, np.full((8, 8), 255, np.uint8))
        self.assertEqual(int(mask.sum()), 8 * 8 * 255)

    def test_v2_context_features_and_model(self):
        image = np.full((20, 28, 3), 235, np.uint8)
        feature_map = _features_v2(image)
        self.assertEqual(feature_map.shape, (20, 28, V2_FEATURES))
        weights = {"w1": np.zeros((V2_FEATURES, 4), np.float32), "b1": np.zeros(4, np.float32),
                   "w2": np.zeros((4, 1), np.float32), "b2": np.zeros(1, np.float32)}
        self.assertEqual(WeakInkV2Model(weights).predict_proba(image).shape, (20, 28))


if __name__ == '__main__': unittest.main()
