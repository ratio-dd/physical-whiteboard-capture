import base64
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from server import WhiteboardApp, make_server  # noqa: E402


class HttpServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = WhiteboardApp(self.temp.name)
        self.httpd = make_server(self.app, port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.history.close()
        self.temp.cleanup()

    def request(self, method, path, body=None, content_type="application/json", headers=None):
        req = urllib.request.Request(self.base + path, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", content_type)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read())

    def test_create_frame_events_and_stop(self):
        status, created = self.request("POST", "/api/sessions", json.dumps({
            "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "width": 320, "height": 240,
        }).encode())
        self.assertEqual(status, 201)
        sid = created["id"]
        def send(frame, stamp):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
            self.assertTrue(ok)
            return self.request("POST", f"/api/sessions/{sid}/frames?t={stamp}", encoded.tobytes(), "image/jpeg")[1]

        blank = np.full((240, 320, 3), 245, np.uint8)
        for i in range(6):
            result = send(blank, 1000 + i * 100)
        self.assertEqual(result["held"], 0.0)
        self.assertGreaterEqual(result["glare"], 0.0)
        self.assertLessEqual(result["glare"], 1.0)
        blank_board = base64.b64decode(result["board"])
        cached = send(blank, 2000)
        # The browser already owns this exact sequence, so the service omits
        # the two unchanged PNG payloads while retaining live indicators.
        ok, encoded = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 96])
        self.assertTrue(ok)
        status, cached = self.request("POST", f"/api/sessions/{sid}/frames?t=2100&seen_seq={result['seq']}", encoded.tobytes(), "image/jpeg")
        self.assertEqual(cached["seq"], result["seq"])
        self.assertNotIn("board", cached)
        self.assertNotIn("unknown", cached)
        self.assertIn("held", cached)
        self.assertIn("glare", cached)
        writing = blank.copy()
        cv2.line(writing, (40, 80), (270, 160), (0, 0, 255), 6)
        for i in range(6):
            result = send(writing, 3000 + i * 100)
        self.assertGreater(result["seq"], 0)
        written_board = base64.b64decode(result["board"])
        self.assertNotEqual(blank_board, written_board)
        _, events = self.request("GET", f"/api/sessions/{sid}/events")
        self.assertGreaterEqual(len(events["events"]), 1)
        self.assertEqual([event["seq"] for event in events["events"]], list(range(1, len(events["events"]) + 1)))
        _, ink = self.request("GET", f"/api/sessions/{sid}/ink?seq={result['seq']}")
        self.assertEqual(ink["version"], 1)
        self.assertEqual(ink["session_id"], sid)
        self.assertEqual(ink["through_seq"], result["seq"])
        self.assertIsInstance(ink["events"], list)
        self.request("POST", f"/api/sessions/{sid}/stop", b"{}")
        self.request("POST", f"/api/sessions/{sid}/stop", b"{}")
        req = urllib.request.Request(self.base + f"/api/sessions/{sid}/frames?t=5000", data=b"bad", method="POST")
        req.add_header("Content-Type", "image/jpeg")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        self.assertEqual(caught.exception.code, 400)

    def test_history_survives_service_restart(self):
        _, created = self.request("POST", "/api/sessions", json.dumps({
            "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "width": 320, "height": 240,
        }).encode())
        sid = created["id"]
        frame = np.full((240, 320, 3), 245, np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        self.assertTrue(ok)
        for i in range(6):
            self.request("POST", f"/api/sessions/{sid}/frames?t={1000 + i * 100}", encoded.tobytes(), "image/jpeg")
        writing = frame.copy()
        cv2.line(writing, (40, 80), (270, 160), (0, 0, 255), 6)
        ok, encoded = cv2.imencode(".jpg", writing, [cv2.IMWRITE_JPEG_QUALITY, 96])
        self.assertTrue(ok)
        last = None
        for i in range(6):
            _, last = self.request("POST", f"/api/sessions/{sid}/frames?t={3000 + i * 100}", encoded.tobytes(), "image/jpeg")
        self.assertGreater(last["seq"], 0)
        _, before = self.request("GET", f"/api/sessions/{sid}/state")
        _, events_before = self.request("GET", f"/api/sessions/{sid}/events")
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.history.close()
        restarted_app = WhiteboardApp(self.temp.name)
        restarted = make_server(restarted_app, port=0)
        thread = threading.Thread(target=restarted.serve_forever, daemon=True)
        thread.start()
        old_base = self.base
        self.base = f"http://127.0.0.1:{restarted.server_port}"
        try:
            status, result = self.request("GET", f"/api/sessions/{sid}/state")
            self.assertEqual(status, 200)
            self.assertEqual(result["seq"], before["seq"])
            self.assertEqual(result["board"], before["board"])
            self.assertEqual(result["unknown"], before["unknown"])
            _, events_after = self.request("GET", f"/api/sessions/{sid}/events")
            self.assertEqual(events_after["events"], events_before["events"])
            self.assertEqual(events_after["events"][-1]["seq"], result["seq"])
            _, segment = self.request("GET", f"/api/sessions/{sid}/replay?seq={result['seq']}")
            self.assertLessEqual(segment["start"], result["seq"])
            self.assertGreaterEqual(segment["end"], result["seq"])
            self.assertIn("board", segment["snapshot"])
            self.assertIn("known", segment["snapshot"])
            self.assertTrue(all("seq" in frame and "patches" in frame for frame in segment["frames"]))
        finally:
            restarted.shutdown()
            restarted.server_close()
            restarted_app.history.close()
            self.base = old_base

    def test_rejects_bad_dimensions_and_origin(self):
        bad = json.dumps({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "width": True, "height": 24}).encode()
        req = urllib.request.Request(self.base + "/api/sessions", data=bad, method="POST")
        req.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        self.assertEqual(caught.exception.code, 400)
        req = urllib.request.Request(self.base + "/api/sessions", method="GET", headers={"Origin": "http://127.0.0.1:9999"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        self.assertEqual(caught.exception.code, 403)

    def test_slow_ink_is_pending_and_duplicate_requests_share_job(self):
        _, created = self.request("POST", "/api/sessions", json.dumps({
            "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "width": 320, "height": 240,
        }).encode())
        sid = created["id"]
        import ink
        original = ink.build_ink
        calls = []
        def slow(history, session_id, upto_seq=None):
            calls.append(1)
            time.sleep(0.7)
            return {"version": 1, "session_id": session_id, "width": 320, "height": 240, "through_seq": 0, "events": []}
        try:
            ink.build_ink = slow
            responses = []
            def fetch():
                req = urllib.request.Request(self.base + f"/api/sessions/{sid}/ink?seq=0")
                with urllib.request.urlopen(req) as response:
                    responses.append((response.status, json.loads(response.read())))
            threads = [threading.Thread(target=fetch), threading.Thread(target=fetch)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual([status for status, _ in responses], [202, 202])
            self.assertEqual(len(calls), 1)
            status = 202
            result = None
            for _ in range(10):
                time.sleep(0.2)
                status, result = self.request("GET", f"/api/sessions/{sid}/ink?seq=0")
                if status == 200:
                    break
            self.assertEqual(status, 200)
            self.assertEqual(result["version"], 1)
        finally:
            ink.build_ink = original

    def test_precomputed_yolo_package_is_opt_in_and_session_bound(self):
        session = self.app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 320, 240)
        package_dir = Path(self.temp.name) / "replay-packages" / session.id
        package_dir.mkdir(parents=True)
        package = {
            "version": 1, "session_id": session.id, "session_version": 0,
            "through_seq": 0, "width": 320, "height": 240,
            "recognition_mode": "yolo-experiment", "model_version": "fixture-yolo",
            "weights_sha256": "fixture", "events": [],
        }
        (package_dir / "0.json").write_text(json.dumps(package), encoding="utf-8")
        metal = dict(package, recognition_mode="yolo-mps-experiment", model_version="fixture-metal")
        (package_dir / "0.yolo-mps-experiment.json").write_text(json.dumps(metal), encoding="utf-8")
        _, directory = self.request("GET", f"/api/sessions/{session.id}/ink-packages")
        self.assertEqual({item["mode"] for item in directory["packages"]}, {"yolo-experiment", "yolo-mps-experiment"})
        _, result = self.request("GET", f"/api/sessions/{session.id}/ink?mode=yolo-experiment")
        self.assertEqual(result["recognition_mode"], "yolo-experiment")
        self.assertEqual(result["weights_sha256"], "fixture")
        _, result = self.request("GET", f"/api/sessions/{session.id}/ink?mode=yolo-mps-experiment")
        self.assertEqual(result["recognition_mode"], "yolo-mps-experiment")
        self.assertEqual(result["model_version"], "fixture-metal")
        color = dict(package, recognition_mode="source-color-experiment", model_version="fixture-faint-color")
        (package_dir / "0.source-color-experiment.json").write_text(json.dumps(color), encoding="utf-8")
        lecturenet = dict(package, recognition_mode="lecturenet-experiment", model_version="LectureNet-V34-FCN-source-evidence-v1", recognition_threshold=0.5, weights_sha256="d3721339")
        (package_dir / "0.lecturenet-experiment.json").write_text(json.dumps(lecturenet), encoding="utf-8")
        _, directory = self.request("GET", f"/api/sessions/{session.id}/ink-packages")
        self.assertEqual({item["mode"] for item in directory["packages"]}, {
            "yolo-experiment", "yolo-mps-experiment", "source-color-experiment", "lecturenet-experiment"})
        _, result = self.request("GET", f"/api/sessions/{session.id}/ink?mode=source-color-experiment")
        self.assertEqual(result["recognition_mode"], "source-color-experiment")
        self.assertEqual(result["model_version"], "fixture-faint-color")
        _, result = self.request("GET", f"/api/sessions/{session.id}/ink?mode=lecturenet-experiment")
        self.assertEqual(result["recognition_mode"], "lecturenet-experiment")
        self.assertEqual(result["model_version"], "LectureNet-V34-FCN-source-evidence-v1")
        self.assertEqual(result["recognition_threshold"], 0.5)
        package["session_version"] = 1
        (package_dir / "0.json").write_text(json.dumps(package), encoding="utf-8")
        metal["session_version"] = 1
        (package_dir / "0.yolo-mps-experiment.json").write_text(json.dumps(metal), encoding="utf-8")
        color["session_version"] = 1
        (package_dir / "0.source-color-experiment.json").write_text(json.dumps(color), encoding="utf-8")
        lecturenet["session_version"] = 1
        (package_dir / "0.lecturenet-experiment.json").write_text(json.dumps(lecturenet), encoding="utf-8")
        _, directory = self.request("GET", f"/api/sessions/{session.id}/ink-packages")
        self.assertEqual(directory["packages"], [])


if __name__ == "__main__":
    unittest.main()
