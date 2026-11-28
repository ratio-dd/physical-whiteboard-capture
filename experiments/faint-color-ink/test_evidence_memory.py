import sys
from pathlib import Path
import unittest
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_memory import EvidenceMemory, MemoryConfig, canonical_raw_mask

def base():
    return np.full((32, 48, 3), (235, 245, 247), np.uint8), np.full((32, 48), 255, np.uint8)

class EvidenceMemoryTests(unittest.TestCase):
  def test_canonical_raw_mask_is_score_threshold_plus_known_only(self):
    score = np.array([[.8, .79], [.9, 0]], np.float32)
    known = np.array([[255, 255], [0, 255]], np.uint8)
    np.testing.assert_array_equal(canonical_raw_mask(score, known), [[255, 0], [0, 0]])

  def test_rescues_only_known_source_supported_pixels(self):
    b, k = base(); s = np.zeros(k.shape, np.float32); s[15, 20] = .9
    src = np.zeros(k.shape, np.uint8); src[15, 20] = 255
    m = EvidenceMemory(); self.assertTrue(m.update(b,k,s,src)[15,20])
    dropped = m.update(b,k,np.zeros_like(s),src); self.assertTrue(dropped[15,20])
    k[:, :24] = 0; self.assertFalse(m.update(b,k,np.zeros_like(s),src)[15,20])

  def test_changed_colour_does_not_leave_residue_and_reflection_without_source_does_not_rescue(self):
    b, k = base(); s=np.zeros(k.shape,np.float32); s[15,20]=.9; src=np.zeros(k.shape,np.uint8);src[15,20]=255
    m=EvidenceMemory(MemoryConfig(colour_tolerance=4)); m.update(b,k,s,src)
    changed=b.copy(); changed[15,20]=(20,20,20); self.assertFalse(m.update(changed,k,np.zeros_like(s),src)[15,20])
    m.reset(); m.update(b,k,s,src); self.assertFalse(m.update(b,k,np.zeros_like(s),np.zeros_like(src))[15,20])

if __name__ == '__main__': unittest.main()
