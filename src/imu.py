import numpy as np

class IMUIntegrator:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

    def integrate(self, accel, dt):
        self.velocity += accel * dt
        self.position += self.velocity * dt
        return self.position.copy()
