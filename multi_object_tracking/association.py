import numpy as np
import scipy.spatial

CANDIDATE_PIXEL_RADIUS = 60


def associate(filters, observations):
    """
    Takes in collection of filters and a list of observations,
    performs association of observations to filters (might be many-to-many)
    and returns a list of associations (filter to list-of-observations)
    and a list of unassociated observations.

    Some filters might be observed multiple times. Unassociated oservations
    will turn into new filters.
    """
    if len(filters) == 0:
        return [], enumerate(observations)
    if len(observations) == 0:
        return [], []

    unassociated_observations = set(range(len(observations)))
    associations = []
    filter_tree = scipy.spatial.cKDTree(
        np.array([flt.mean() for flt in filters]))
    observation_tree = scipy.spatial.cKDTree(
        np.array([obs.centroid for obs in observations]))
    candidates = filter_tree.query_ball_tree(observation_tree, CANDIDATE_PIXEL_RADIUS)

    for i, flt in enumerate(filters):
        obs_candidates = [(j, observations[j]) for j in candidates[i]]
        best_observation = find_best_observation(flt, obs_candidates)
        if best_observation is None:
            continue
        j, obs = best_observation
        unassociated_observations.discard(j)
        associations.append((flt, (j, obs)))

    unassociated_observations = [(j, observations[j]) for j in unassociated_observations]
    return associations, unassociated_observations


def find_best_observation(flt, observations):
    def dist(j_obs):
        _, obs = j_obs
        return flt.dist(obs)

    distances = [(o, dist(o)) for o in observations]
    distances = [(o, d) for o, d in distances
                 if d < 10]

    if not distances:
        return None

    closest = min(distances, key=lambda od: od[1])[0]
    return closest
