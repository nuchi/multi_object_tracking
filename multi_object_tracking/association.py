import numpy as np
import scipy.spatial

from . import params


def associate(filters, observations):
    """
    Takes in collection of filters and a list of observations, performs association
    of observations to filters and returns a list of tuples (filter, (index, observation))
    and a list of unassociated observations.

    The indices are useful for debugging, when it's helpful to visualize which observation
    corresponds to which filter.
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

    # The following candidates use a large radius (default 60 pixels) just to prune down
    # the search space. A more careful pruning is done within `find_best_observation`.
    candidates = filter_tree.query_ball_tree(
        observation_tree, params.ASSOC_CANDIDATE_PIXEL_RADIUS)

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
        # flt.dist is negative log likelihood
        return flt.dist(obs)

    distances = [(o, dist(o)) for o in observations]
    distances = [(o, d) for o, d in distances
                 if d < params.ASSOC_LOG_LKL_THRESHOLD]

    if not distances:
        return None

    closest = min(distances, key=lambda od: od[1])[0]
    return closest
