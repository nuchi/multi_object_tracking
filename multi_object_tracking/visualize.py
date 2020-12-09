import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arrow
from matplotlib.collections import PatchCollection
import numpy as np


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[0, 1], vecs[0, 0]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, lw=0.5, **kwargs)
    return ellip


def plot_covariances(observation_list, **kwargs):
    ellipses = [
        plot_cov_ellipse(o['cov'], o['centroid'], facecolor='none', edgecolor=o['color'], **kwargs)
        for o in observation_list
    ]
    collection = PatchCollection(ellipses, match_original=True)
    return collection


def plot_observation_links(observation_list, filters):
    arrows = []
    for f in filters.values():
        if f['last_observed'] == 0:
            obs_centroid = observation_list[f['last_observation']]['centroid']
            diff = [f['x'][:2][i] - obs_centroid[i] for i in (0, 1)]
            arrows.append(
                Arrow(x=obs_centroid[0], y=obs_centroid[1], dx=diff[0], dy=diff[1],
                      edgecolor='b', facecolor='none', lw=0.5))
    return PatchCollection(arrows, match_original=True)


def get_color(f):
    if f['is_duplicate']:
        return 'orange'
    if f['last_observed'] == 3:
        return 'red'
    if f['age'] == 0:
        return 'lightgreen'
    return 'yellow'


def plot_debug_data(debug_data, n, ax=None):
    ax = ax or plt.gca()
    current_collections = [c for c in ax.collections]
    for coll in current_collections:
        coll.remove()

    observations = debug_data[n]['observations']
    filters = dict(debug_data[n]['filters'], **debug_data[n]['invalid_filters'])

    # Observations
    obs_collection = plot_covariances(
        [{**o, 'color': 'b'} for i, o in enumerate(observations)])
    ax.add_collection(obs_collection)

    # Filter centroids
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    filter_centroid_collection = plot_covariances([
        {
            'cov': np.dot(np.dot(A, f['P']), A.T),
            'centroid': f['x'][:2],
            'color': get_color(f),
        }
        for _id, f in filters.items()
    ])
    ax.add_collection(filter_centroid_collection)

    # velocity arrows
    arrow_keys = ['x', 'y', 'dx', 'dy']
    velocity_arrow_collection = PatchCollection([
        Arrow(**dict(zip(arrow_keys, f['x'])), edgecolor=get_color(f), facecolor='none', lw=0.5)
        for f in filters.values()
    ], match_original=True)
    ax.add_collection(velocity_arrow_collection)

    # velocity covariances
    velocity_cov_collection = plot_covariances([
        {
            'cov': np.array(f['P'])[2:, 2:],
            'centroid': [f['x'][0] + f['x'][2], f['x'][1] + f['x'][3]],
            'color': get_color(f),
        }
        for f in filters.values()
    ], ls=':')
    ax.add_collection(velocity_cov_collection)

    # Observations to filter centroids
    observation_link_collection = plot_observation_links(observations, filters)
    ax.add_collection(observation_link_collection)
