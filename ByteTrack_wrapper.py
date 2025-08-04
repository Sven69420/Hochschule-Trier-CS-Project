import numpy as np
import torch

from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace

# Compatibility fix for deprecated NumPy aliases
if not hasattr(np, 'float'):
    np.float = float

class Tracker:
    def __init__(self, args=None):
        if args is None:
            args = {
                "track_thresh": 0.3,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 20
            }
        args_ns = SimpleNamespace(**args)
        self.tracker = BYTETracker(args_ns, frame_rate=args_ns.frame_rate)

    def update(self, detections, img_info, img_size):
        """
        detections: ndarray (n, 6) with [x1, y1, x2, y2, score, class_id]
        """
        if isinstance(img_info, dict):
            # Convert dict to list for ByteTrack if needed
            img_info_fixed = [img_info.get("height", 0), img_info.get("width", 0)]
        else:
            img_info_fixed = img_info

        # ByteTrack expects a numpy array [N, 5] or [N, 6] on CPU
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()  # Ensure numpy array on CPU

        # Call ByteTrack
        outputs = self.tracker.update(detections, img_info_fixed, img_size)

        # debug output to determine if the tracker is working
        # print(f"BYTETracker.update() returned {len(outputs)} raw tracks")  

        tracks = []
        for track in outputs:
            tlwh = track.tlwh
            tid = track.track_id
            score = track.score
            tracks.append((tid, tlwh, score))
        return tracks

def format_detections_for_tracker(detections):
    """
    Convert detections into format expected by BYTETracker:
    [x, y, w, h, score, class]
    """
    if detections is None or len(detections) == 0:
        return np.empty((0, 6), dtype=np.float32)

    formatted = []
    for det in detections:
        x1, y1, x2, y2, score, cls = det
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            formatted.append([x1, y1, w, h, score, cls])
    return np.array(formatted, dtype=np.float32)



if __name__ == '__main__':
    print("This file only contains helper functions and classes. Run DynamicObjectFiltering.py instead.")