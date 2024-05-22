from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np

class Tracker():
    def __init__(self, id, x, obj, without_obj=0):
        self.id = id
        self.obj = obj
        self.without_obj = without_obj
        self.scaled = False
        self.object_detected = True
        self.kf = KalmanFilter(dim_x = 4, dim_z=2)
        self.kf.x = np.array([x[0], 0, x[1], 0])
        self.kf.P *= 500
        self.kf.R *= 5
        self.kf.F = np.array([
                        [1, 0.1, 0, 0],
                        [0,  1, 0, 0],
                        [0, 0, 1, 0.1],
                        [0, 0, 0, 1]], dtype=float)
        self.kf.H = np.array([
                        [1, 0, 0, 0],
                        [0, 0, 1, 0]
                        ])
        q = Q_discrete_white_noise(dim=2, dt=0.1, var=1)
        self.kf.Q = block_diag(q, q)
    def predict(self):
        return self.kf.predict()
    
    def update(self, measurement):
        self.kf.update(measurement)
