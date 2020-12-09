import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.sparse
import scipy.spatial

from . import params
from .track import track

OFFSETS = {
    1: np.array([0,0]),
    2: np.array([640,0]),
    3: np.array([0,512]),
    4: np.array([640, 512]),
}


class BadVideoException(Exception):
    pass


def load_labels(basename, frame_num):
    labels = []
    found = False
    frame_num = frame_num + 1  # frame number in file name is 1-indexed

    for i in (1, 2, 3, 4):
        filename = f'{basename}_{frame_num:04d}_{i}_labelbasic.json'
        try:
            with open(filename) as f:
                data = json.load(f)
                labels.extend([
                    np.mean(s['points'], axis=0) + OFFSETS[i]
                    for s in data['shapes']
                ])
                found = True
        except FileNotFoundError:
            pass

    return labels, found


def compute_stats(predictions, labels):
    predictions_tree = scipy.spatial.cKDTree(predictions)
    labels_tree = scipy.spatial.cKDTree(labels)

    l_to_ps = labels_tree.query_ball_tree(
        predictions_tree, params.EVAL_MATCH_PIXEL_THRESHOLD)
    indices = []
    indptr = [0]
    for ps in l_to_ps:
        indices.extend(ps)
        indptr.append(len(indices))
    data = np.ones((len(indices),))
    matrix = scipy.sparse.csr_matrix((data, indices, indptr))
    best_matches = scipy.sparse.csgraph.maximum_bipartite_matching(
        matrix, perm_type='column')

    matches = np.count_nonzero(best_matches >= 0)

    recall = matches / len(labels)
    precision = matches / len(predictions)
    fp = len(predictions) - matches
    fn = len(labels) - matches
    return dict(
        matches=matches,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_video(filename, labels_dir, use_filters, start, end):
    labels_base = os.path.join(
        labels_dir,
        os.path.splitext(os.path.basename(filename))[0]
    )

    if start is None or end is None:
        label_names = glob.glob(f'{labels_base}_*_labelbasic.json')
        pat = re.compile('^{prefix}_([0-9]*)_'.format(prefix=re.escape(labels_base)))
        frame_nums = set(
            int(re.match(pat, name).groups()[0], base=10) - 1
            for name in label_names
        )
        if start is None:
            start = max(0, min(frame_nums) - 20)
        if end is None:
            end = max(frame_nums) + 20

        print(f'Auto-detected beginning and end of labels; using start {start}, end {end}')

    if end == -1:
        end = None

    try:
        _, debug_data = track(filename, debug=True, start=start, end=end)
    except Exception as e:
        raise BadVideoException(e)
    end = end or (start + len(debug_data))

    matches = 0
    fp = 0
    fn = 0

    for i in range(start, end):
        labels, found = load_labels(labels_base, i)
        if not found:
            continue
        predictions = [
                f['x'][:2] for f in debug_data[i - start]['filters'].values()
                if not f['is_duplicate'] and f['last_observed'] < 3
            ] \
            if use_filters \
            else [o['centroid'] for o in debug_data[i - start]['observations']]
        frame_stats = compute_stats(
            predictions,
            labels,
        )
        matches += frame_stats['matches']
        fp += frame_stats['false_positives']
        fn += frame_stats['false_negatives']

    precision = float('inf') if (matches + fp) == 0 else matches / (matches + fp)
    recall = float('inf') if (matches + fn) == 0 else matches / (matches + fn)

    return dict(
        matches=matches,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall
    )


def evaluate_all(list_of_videos, use_filters, start, end):
    matches = 0
    false_positives = 0
    false_negatives = 0

    for line in list_of_videos.readlines():
        line = line.strip()
        if not line:
            continue

        try:
            video, labels_dir = line.split(',')
        except ValueError:
            print(f'Bad line, skipping: {line}')
            continue

        print(f'Stats for {os.path.basename(video)}:')
        try:
            stats = evaluate_video(video, labels_dir, use_filters, start, end)
        except BadVideoException as e:
            print(e)
            raise e
            print(f'Bad video {video}, skipping')
            continue

        print(stats)
        if float('inf') in (stats['recall'], stats['precision']):
            print('****** Weird result -- check that the labels directory is correct?')

        matches += stats['matches']
        false_positives += stats['false_positives']
        false_negatives += stats['false_negatives']

    if list_of_videos is not sys.stdin:
        list_of_videos.close()

    print('--------------')

    if max(matches, false_positives, false_negatives) == 0:
        print('No videos or labels found, exiting.')
        return 1

    print('Overall stats:')
    print(f'Matches: {matches}')
    print(f'False positives: {false_positives}')
    print(f'False negatives: {false_negatives}')
    print(f'Precision: {matches / (matches + false_positives)}')
    print(f'Recall: {matches / (matches + false_negatives)}')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate evaluation metrics.')
    parser.add_argument(
        '-i', '--input', default=sys.stdin, type=argparse.FileType(), dest='list_of_videos',
        help='Path to file containing lines like <path-to-video>,<path-to-labels-dir>. '
             'Please ensure the paths do not contain commas.')
    parser.add_argument(
        '--use-filters', action='store_true',
        help='Use kalman filters for evaluation, otherwise just use frame-by-frame observations.')
    parser.add_argument(
        '-s', '--start', default=None, type=int,
        help='Frame number at which to start. Omit to auto-detect based on labels.')
    parser.add_argument(
        '-e', '--end', default=None, type=int,
        help='Frame number at which to stop. Omit to auto-detect based on labels. '
             'Use -1 to go until the end of the video.')
    args, _ = parser.parse_known_args()

    sys.exit(evaluate_all(**vars(args)))
