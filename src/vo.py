import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, K):
        self.K = K
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def update(self, pts_old, pts_new):
        if pts_old is None or len(pts_old) < 8:
            return self.R, self.t
        E, mask = cv2.findEssentialMat(pts_old, pts_new, self.K, method=cv2.RANSAC)
        if E is None:
            return self.R, self.t
        _, R, t, mask = cv2.recoverPose(E, pts_old, pts_new, self.K)
        self.t = self.t + self.R @ t
        self.R = R @ self.R
        return self.R, self.t
