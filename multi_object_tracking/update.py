from collections import defaultdict

import numpy as np
import scipy.spatial

from .kalman_filter import KalmanConstVelFilter


def update(associations):
    for flt, observation in associations:
        flt.update(observation)


def make_new_filters(unassociated_observations):
    return [KalmanConstVelFilter(j_obs) for j_obs in unassociated_observations]


def deduplicate(filters):
    filter_tree = scipy.spatial.cKDTree(
        np.array([f._kf.x.reshape((4,)) for f in filters]))
    nearby_pairs = filter_tree.query_pairs(2.0)
    components = connected_components(nearby_pairs)
    for component in components:
        best = min(
            component,
            key=lambda idx: np.linalg.det(filters[idx]._kf.P))
        for idx in component:
            if idx == best:
                continue
            filters[idx].is_duplicate = True


def connected_components(list_of_pairs):
    nodes = set(n for pair in list_of_pairs for n in pair)
    edges = defaultdict(set)
    for a, b in list_of_pairs:
        edges[a].add(b)
        edges[b].add(a)
    components = []
    while nodes:
        n = nodes.pop()
        component = set([n])
        components.append(component)
        todo = set([n])
        while todo:
            m = todo.pop()
            todo.update(edges[m].intersection(nodes))
            nodes.difference_update(edges[m])
            component.update(edges[m])
            del edges[m]
    return components
