import itertools

import filterpy.common
from filterpy.stats import logpdf
import numpy as np

from . import params


def update_covariance(observation):
    return 0.5 * observation.covariance + np.eye(2)


class KalmanConstVelFilter:
    id_iter = itertools.count()

    def __init__(self, idx_observation, *args, **kwargs):
        kf = filterpy.common.kinematic.kinematic_kf(
            dim=2,
            order=1,
            order_by_dim=False
        )
        idx, observation = idx_observation
        kf.x[:2] = np.reshape(observation.centroid, (2, 1))
        kf.x[2:] = np.array([[0.], [0.]])
        kf.P = np.diag(params.KF_INITIAL_DIAG_COV)
        kf.P[:2, :2] = update_covariance(observation)
        kf.R = np.diag([1.0, 1.0])
        kf.Q = filterpy.common.discretization.Q_discrete_white_noise(
            dim=2, block_size=2, order_by_dim=False, var=3.)
        self._kf = kf
        self.age = 0
        self.last_observed = 0
        self.last_observation = idx
        self.is_duplicate = False
        self.id = next(KalmanConstVelFilter.id_iter)

    def predict(self):
        self._kf.predict()
        self.last_observed += 1
        self.age += 1

    def mean(self):
        return np.squeeze(self._kf.measurement_of_state(self._kf.x))

    def update(self, observation):
        idx, obs = observation
        self._kf.update(obs.centroid, R=update_covariance(obs))
        self.last_observed = 0
        self.last_observation = idx

    def dist(self, observation):
        # negative log likelihood (so that smaller is better)
        kf = self._kf
        residual = observation.centroid - self.mean()
        S = np.dot(kf.H, np.dot(kf.P, kf.H.T)) + observation.covariance
        return -logpdf(x=residual, cov=S)

    def dump(self):
        return {
            'age': self.age,
            'last_observed': self.last_observed,
            'last_observation': self.last_observation,
            'is_duplicate': self.is_duplicate,
            'x': np.squeeze(self._kf.x).tolist(),
            'P': self._kf.P.tolist(),
        }
