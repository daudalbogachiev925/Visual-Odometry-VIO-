import numpy as np

class EKF:
    """Простой EKF для fusion VO + IMU."""
    def __init__(self):
        self.x = np.zeros(9)  # [p, v, bias]
        self.P = np.eye(9) * 0.1
        self.Q = np.eye(9) * 0.01
        self.R = np.eye(3) * 0.5

    def F(self, dt):
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        return F

    def H(self):
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        return H

    def predict(self, accel, dt):
        F = self.F(dt)
        self.x[0:3] += self.x[3:6] * dt + 0.5 * accel * dt**2
        self.x[3:6] += accel * dt
        self.P = F @ self.P @ F.T + self.Q
        return self.x[0:3]

    def update(self, z_pos):
        H = self.H()
        y = z_pos - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        return self.x[0:3]
