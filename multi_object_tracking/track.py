import itertools

import tqdm

from .update import update, make_new_filters, deduplicate
from .association import associate
from .observation import observations_from_frame
from .preprocess import foreground, stream_video


def track(filename, debug=False, start=0, end=None):
    frames = foreground(
        itertools.islice(stream_video(filename), start, end),
        20,
    )

    filters = []
    filter_counts = []

    # Returned, but only populated if debug is True
    debug_data = []

    for frame in tqdm.tqdm(frames):
        # Generate observations
        observations = observations_from_frame(frame)

        # Associate observations to existing filters
        for f in filters:
            f.predict()
        associations, unassociated_observations = associate(filters, observations)

        # Update filters with their corresponding observations
        update(associations)

        # Make new filters for unassociated observations
        new_filters = make_new_filters(unassociated_observations)
        filters.extend(new_filters)

        # If we have multiple filters tracking the same underlying object, mark
        # the redundant ones as being duplicates.
        filters, duplicates = deduplicate(filters)
        # Mark duplicates for debugging purposes
        for f in duplicates:
            f.is_duplicate = True

        # Remove stale filters
        valid_filters = []
        stale_filters = []
        for f in filters:
            valid_filters.append(f) if f.last_observed < 3 else stale_filters.append(f)
        filters = valid_filters

        # Add debug information
        if debug:
            debug_data.append({
                'observations': [
                    {
                        'centroid': o.centroid.tolist(),
                        'cov': o.covariance.tolist()
                    }
                    for o in observations
                ],
                'filters': {
                    f.id: f.dump()
                    for f in filters
                },
                'invalid_filters': {
                    f.id: f.dump()
                    for f in duplicates + stale_filters
                }
            })

        filter_counts.append(len(filters))

    return filter_counts, debug_data
