import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lecturemath-ink"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "faint-color-ink"))
from color_detector import ColorState, ColorConfig, detect

def test_first_frame_is_reference_and_synthetic_red_is_exportable():
    board = np.full((100, 160, 3), (235, 245, 247), np.uint8)
    known = np.full((100, 160), 255, np.uint8)
    state = ColorState(); first = detect(board, known, state)
    assert not first.mask.any()
    marked = board.copy(); cv2.line(marked, (20, 50), (130, 50), (180, 205, 245), 2)
    result = detect(marked, known, state, ColorConfig(local_threshold=2, score_threshold=2, chroma_threshold=2, ridge_threshold=1, delta_threshold=.1, ridge_kernel=5, temporal_confirm=1))
    assert result.mask.any() and result.paths
    assert all(p["kind"] == "model_candidate" for p in result.paths)

def test_fixed_brightness_band_is_not_colour_ink():
    board = np.full((100, 160, 3), (235, 245, 247), np.uint8)
    known = np.full((100, 160), 255, np.uint8); state = ColorState(); detect(board, known, state)
    glare = board.copy(); glare[20:27] = (250, 250, 250)
    result = detect(glare, known, state, ColorConfig(local_threshold=2, score_threshold=2, chroma_threshold=2, ridge_threshold=1))
    assert not result.mask.any()

def test_unknown_pixels_never_become_candidates():
    board = np.full((60, 80, 3), (235, 245, 247), np.uint8); known = np.full((60,80),255,np.uint8)
    state = ColorState(); detect(board, known, state); marked=board.copy(); cv2.line(marked,(5,30),(70,30),(180,205,245),2)
    known[:, :40] = 0; result=detect(marked,known,state,ColorConfig(local_threshold=2,score_threshold=2,chroma_threshold=2,ridge_threshold=1))
    assert not (result.mask[:, :40]).any()

def test_repeated_identical_frame_does_not_erase_visible_ink():
    board=np.full((80,120,3),(235,245,247),np.uint8); cv2.line(board,(10,40),(100,40),(180,205,245),2); known=np.full((80,120),255,np.uint8); state=ColorState()
    first=detect(board,known,state); second=detect(board,known,state)
    assert first.mask.any() and second.mask.any()

def test_occlusion_unknown_then_reappearance_reacquires_without_replaying_unknown():
    board=np.full((80,120,3),(235,245,247),np.uint8); cv2.line(board,(10,40),(100,40),(180,205,245),2); visible=np.full((80,120),255,np.uint8); state=ColorState(); detect(board,visible,state)
    hidden=np.zeros_like(visible); assert not detect(board,hidden,state).mask.any()
    again=detect(board,visible,state); assert again.mask.any()

def test_unrelated_patch_is_not_recolored_and_erasure_clears():
    board=np.full((80,120,3),(235,245,247),np.uint8); cv2.line(board,(10,40),(100,40),(180,205,245),2); k=np.full((80,120),255,np.uint8); state=ColorState(); detect(board,k,state)
    blank=np.full_like(board,(235,245,247)); assert not detect(blank,k,state).mask.any()
    other=blank.copy(); cv2.line(other,(10,65),(100,65),(180,205,245),2); result=detect(other,k,state); assert result.mask.any(); assert not result.mask[:55].any()
