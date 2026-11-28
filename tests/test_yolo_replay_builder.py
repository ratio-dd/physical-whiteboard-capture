import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "yolo-ink"))
from server import WhiteboardApp  # noqa: E402
from build_replay import build_package  # noqa: E402


class FixturePredictor:
    model_version = "fixture-yolo"

    def predict(self, board):
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        return (gray < 220).astype(np.float32)


class YoloReplayBuilderTest(unittest.TestCase):
    def test_builds_atomic_versioned_package_from_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app = WhiteboardApp(root / "data")
            try:
                session = app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 128, 96)
                board = np.full((96, 128, 3), 245, np.uint8)
                cv2.line(board, (12, 20), (110, 70), (40, 40, 40), 3)
                known = np.full((96, 128), 255, np.uint8)
                app.history.record(session.id, board, known, 1000, 1000)
                destination = build_package(root / "data", session.id, root / "packages",
                                            FixturePredictor(), "fixture-yolo", "fixture-sha", .7616142630577087)
                package = json.loads(destination.read_text())
                self.assertEqual(package["recognition_mode"], "yolo-experiment")
                self.assertEqual(package["session_version"], 1)
                self.assertEqual(package["through_seq"], 1)
                self.assertEqual(package["weights_sha256"], "fixture-sha")
                self.assertEqual(package["recognition_threshold"], .7616142630577087)
                self.assertTrue(package["events"])
                self.assertFalse(list(destination.parent.glob("*.tmp-*")))
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()
