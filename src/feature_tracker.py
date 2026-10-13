import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, max_features=300):
        self.max_features = max_features
        self.prev_pts = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def track(self, prev_gray, gray):
        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, self.max_features, 0.01, 10)
            return None, None
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, self.prev_pts, None, **self.lk_params)
        status = status.flatten()
        good_new = next_pts[status == 1]
        good_old = self.prev_pts[status == 1]
        self.prev_pts = good_new.reshape(-1, 1, 2)
        return good_old, good_new
