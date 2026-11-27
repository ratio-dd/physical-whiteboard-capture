"""Run with WHITEBOARD_BROWSER_TESTS=1 and an installed Playwright CLI/Chromium."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import uuid

import numpy as np

from server import WhiteboardApp, make_server


@unittest.skipUnless(os.environ.get("WHITEBOARD_BROWSER_TESTS") == "1",
                     "set WHITEBOARD_BROWSER_TESTS=1 to exercise Chromium")
class PlaybackBrowserTest(unittest.TestCase):
    def test_browser_clock_and_session_lifecycle(self):
        cli = shutil.which("playwright-cli")
        self.assertTrue(cli, "install playwright-cli and Chromium for this optional test")
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="whiteboard-playback-"))
        app = WhiteboardApp(directory)
        self.addCleanup(app.history.close)
        board = np.full((32, 40, 3), 248, np.uint8)
        known = np.zeros(board.shape[:2], np.uint8)
        created = 1_700_000_000_000
        sid = app.history.create(board, known, created)
        for seq in range(1, 41):
            y, x = divmod(seq * 3, board.shape[1])
            board[y, x] = (seq, 120, 240 - seq)
            known[y, x] = 255
            confirmed = created + seq * 1000
            self.assertEqual(app.history.record(sid, board, known, confirmed - 200, confirmed), seq)
        app.history.create(board, np.zeros_like(known), created + 60_000)

        http = self.enterContext(make_server(app, port=0))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(http.shutdown)
        url = f"http://127.0.0.1:{http.server_port}/#history"
        session = "playback-test-" + uuid.uuid4().hex
        script = Path(__file__).resolve().with_name("playback-browser.js")

        def run(*args):
            result = subprocess.run([cli, "-s=" + session, *args], cwd=directory,
                                    text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        try:
            run("open", url, "--browser=chromium")
            output = run("run-code", "--filename=" + str(script), "--raw")
            self.assertIn('"passed":true', output)
        finally:
            run("close")


if __name__ == "__main__":
    unittest.main()
