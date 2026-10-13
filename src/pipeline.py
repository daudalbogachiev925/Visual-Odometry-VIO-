import cv2
from src.feature_tracker import FeatureTracker
from src.vo import VisualOdometry
from src.ekf import EKF

class VIOPipeline:
    def __init__(self, K):
        self.tracker = FeatureTracker()
        self.vo = VisualOdometry(K)
        self.ekf = EKF()
        self.prev_gray = None

    def process(self, frame, accel, dt):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            old, new = self.tracker.track(self.prev_gray, gray)
            self.vo.update(old, new)
            self.ekf.predict(accel, dt)
            pos = self.ekf.update(self.vo.t.flatten())
        else:
            pos = self.ekf.x[0:3]
        self.prev_gray = gray
        return pos
