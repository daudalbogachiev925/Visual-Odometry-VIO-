import numpy as np
from src.ekf import EKF

def test_ekf_update_pulls_towards_measurement():
    ekf = EKF()
    ekf.predict(np.zeros(3), 0.1)
    p = ekf.update(np.array([1.0, 0.0, 0.0]))
    assert p[0] > 0
