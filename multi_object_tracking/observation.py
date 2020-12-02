from collections import namedtuple

import cv2 as cv
import numpy as np
import scipy

THRESHOLD = 25
Observation = namedtuple('Observation', ('centroid', 'covariance'))


def observations_from_frame(frame, THRESHOLD=THRESHOLD):
    thresholded = (frame > THRESHOLD).astype(np.uint8)
    cleaned = scipy.ndimage.binary_opening(thresholded).astype(np.uint8)
    n, component_image, stats, centroids = cv.connectedComponentsWithStats(cleaned)
    indices = np.indices(component_image.shape).T[:, :, [1, 0]]
    components = [indices[component_image.T == i] for i in range(1, n)]
    covariances = [np.cov(component, rowvar=False) for component in components]

    observations = [
        Observation(centroids[i + 1][::1], covariance)
        for i, (component, covariance) in enumerate(zip(components, covariances))
    ]
    observations = [
        split_o for obs in observations for split_o in maybe_split(obs)
    ]
    return observations


def maybe_split(observation):
    evals, evecs = np.linalg.eigh(observation.covariance)
    if evals[0] == 0.0:
        return []
    if evals[1] < 20:
        return [observation]

    centroid = observation.centroid
    cov = np.dot(np.dot(evecs.T, np.diag([evals[0], 0.25 * evals[1]])), evecs)
    diff = evecs[1] * np.sqrt(evals[1])
    return [
        Observation(centroid + diff, cov),
        Observation(centroid - diff, cov),
    ]
