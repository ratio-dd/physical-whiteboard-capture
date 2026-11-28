from pathlib import Path
import unittest


class InkSvgSourceTest(unittest.TestCase):
    def test_closed_ink_is_an_outline_and_export_drops_selection(self):
        source = (Path(__file__).parents[1] / "web" / "ink-view.js").read_text()
        self.assertIn("path.setAttribute('fill', 'none')", source)
        self.assertIn("copy.querySelectorAll('.selected').forEach(node => node.classList.remove('selected'))", source)
        self.assertIn("recognition_mode: loadedMode", source)
        self.assertIn("model_version: loadedModelVersion", source)
        self.assertIn("yolo-experiment", source)
        self.assertNotIn("path.setAttribute('fill', stroke.closed", source)


if __name__ == '__main__':
    unittest.main()
