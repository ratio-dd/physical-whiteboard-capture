import tempfile
import time
import unittest
import json
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from live_ink.runtime import OrderedInkRuntime, RuntimeResult
from live_ink.session import LiveSessionStore
from server import WhiteboardApp, make_server


class _Backend:
    def predict(self, bgr):
        return np.zeros(bgr.shape[:2], np.float32)


class _Pipeline:
    def step(self, bgr, known, score):
        return type("Result", (), {"vector_paths": []})()


class LiveSessionTests(unittest.TestCase):
    def test_explicit_mode_persists_svg_and_survives_app_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            def runtime_factory(sink):
                return OrderedInkRuntime(_Backend(), _Pipeline,
                                         lambda result, w, h: f'<svg width="{w}" height="{h}"/>',
                                         result_sink=sink)
            app = WhiteboardApp(temp, live_runtime_factory=runtime_factory)
            session = app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 64, 48, "lecturenet-live")
            frame = np.full((48, 64, 3), 245, np.uint8)
            cv2.line(frame, (8, 8), (56, 36), (0, 0, 0), 2)
            ok, encoded = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            for stamp in range(1000, 1600, 100):
                app.process(session, encoded.tobytes(), stamp)
            for _ in range(50):
                if app.live_ink(session.id).get("seq") == 1: break
                time.sleep(.01)
            status = app.live_ink(session.id)
            self.assertEqual(status["seq"], 1)
            self.assertIn("<svg", status["svg"])
            app.close()
            restarted = WhiteboardApp(temp)
            persisted = restarted.live_ink(session.id)
            self.assertEqual(persisted["seq"], 1)
            self.assertIn("<svg", persisted["svg"])
            restarted.close()

    def test_live_mode_without_factory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            app = WhiteboardApp(temp)
            with self.assertRaises(RuntimeError):
                app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 64, 48, "lecturenet-live")
            app.close()

    def test_http_contract_reports_no_svg_until_new_seq_then_persists(self):
        with tempfile.TemporaryDirectory() as temp:
            def runtime_factory(sink):
                return OrderedInkRuntime(_Backend(), _Pipeline,
                                         lambda result, w, h: f'<svg width="{w}" height="{h}"/>',
                                         result_sink=sink)
            app = WhiteboardApp(temp, live_runtime_factory=runtime_factory)
            httpd = make_server(app, port=0); thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
            base = f"http://127.0.0.1:{httpd.server_port}"
            try:
                payload = json.dumps({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "width": 64, "height": 48,
                                      "recognition_mode": "lecturenet-live"}).encode()
                req = urllib.request.Request(base + "/api/sessions", data=payload, method="POST", headers={"Content-Type": "application/json"})
                sid = json.loads(urllib.request.urlopen(req).read())["id"]
                frame = np.full((48, 64, 3), 245, np.uint8); cv2.line(frame, (8, 8), (56, 36), (0, 0, 0), 2)
                ok, encoded = cv2.imencode(".jpg", frame); self.assertTrue(ok)
                for stamp in range(1000, 1600, 100):
                    req = urllib.request.Request(base + f"/api/sessions/{sid}/frames?t={stamp}", data=encoded.tobytes(), method="POST", headers={"Content-Type": "image/jpeg"})
                    urllib.request.urlopen(req).read()
                status = None
                for _ in range(50):
                    status = json.loads(urllib.request.urlopen(base + f"/api/sessions/{sid}/live-ink?after_seq=0").read())
                    if status.get("seq", 0) > 0: break
                    time.sleep(.01)
                self.assertGreater(status["seq"], 0); self.assertIn("svg", status); self.assertEqual(status["status"], "running")
                unchanged = json.loads(urllib.request.urlopen(base + f"/api/sessions/{sid}/live-ink?after_seq={status['seq']}").read())
                self.assertNotIn("svg", unchanged); self.assertFalse(unchanged["updated"])
                urllib.request.urlopen(urllib.request.Request(base + f"/api/sessions/{sid}/stop", data=b"{}", method="POST")).read()
                package = json.loads(urllib.request.urlopen(base + f"/api/sessions/{sid}/ink?mode=lecturenet-live").read())
                self.assertEqual(package["recognition_mode"], "lecturenet-live")
                self.assertGreater(package["through_seq"], 0)
                self.assertEqual(len(package["events"]), package["through_seq"])
            finally:
                httpd.shutdown(); httpd.server_close(); app.close()

    def test_out_of_order_submit_waits_for_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            def runtime_factory(sink):
                return OrderedInkRuntime(_Backend(), _Pipeline,
                                         lambda result, w, h: '<svg/>', result_sink=sink)
            app = WhiteboardApp(temp, live_runtime_factory=runtime_factory)
            session = app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 16, 12, "lecturenet-live")
            board = np.zeros((12, 16, 3), np.uint8); known = np.full((12, 16), 255, np.uint8)
            errors = []
            def send(seq):
                try: session.live_session.submit(seq, seq, board, known)
                except Exception as exc: errors.append(exc)
            second = threading.Thread(target=send, args=(2,)); second.start(); time.sleep(.03)
            first = threading.Thread(target=send, args=(1,)); first.start(); first.join(2); second.join(2)
            self.assertFalse(errors); self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(session.live_session.runtime.stats().submitted, 2)
            app.close()

    def test_event_storage_is_per_sequence_not_quadratic_history_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            store = LiveSessionStore(temp, "fixture", 16, 12, "lecturenet-live", "fixture", "sha", "proposal_text")
            for seq in range(1, 201):
                store.write_result(RuntimeResult("fixture", seq, seq, "<svg/>",
                    type("Result", (), {"vector_paths": []})(), 1.0))
            self.assertEqual(len(list((Path(temp) / "fixture" / "events").glob("seq-*.json"))), 200)
            self.assertFalse((Path(temp) / "fixture" / "events.json").exists())
            package = store.package(200)
            self.assertEqual(package["through_seq"], 200)
            self.assertEqual(len(package["events"]), 200)

    def test_storage_failure_is_not_reported_as_stopped_success(self):
        with tempfile.TemporaryDirectory() as temp:
            def runtime_factory(sink):
                return OrderedInkRuntime(_Backend(), _Pipeline, lambda result, w, h: '<svg/>', result_sink=sink)
            app = WhiteboardApp(temp, live_runtime_factory=runtime_factory)
            session = app.create([[0, 0], [1, 0], [1, 1], [0, 1]], 16, 12, "lecturenet-live")
            def fail_write(path, value): raise OSError("disk full fixture")
            session.live_session.store._write = fail_write
            session.live_session.submit(1, 1, np.zeros((12, 16, 3), np.uint8), np.full((12, 16), 255, np.uint8))
            for _ in range(50):
                if session.live_session.runtime.stats().state == "failed": break
                time.sleep(.01)
            self.assertEqual(session.live_session.runtime.stats().state, "failed")
            with self.assertRaises(RuntimeError): session.live_session.stop()
            app.close()


if __name__ == "__main__":
    unittest.main()
