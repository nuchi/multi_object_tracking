import argparse
import glob
import json
import os

import tqdm

EXTS = ('mp4', 'avi', 'mov', 'mpeg', 'flv', 'wmv')


parser = argparse.ArgumentParser()
parser.add_argument(
    '-p', '--path', default='videos',
    help='Path to directory containing videos.')
parser.add_argument(
    '-o', '--out', default='output.txt',
    help='Path to output the object counts per video.')
parser.add_argument(
    '-d', '--debug-out',
    help='If present, write debug output to this directory.')
parser.add_argument(
    '-s', '--start', default=0, type=int,
    help='The frame number at which to start. (Start at the beginning by default.)')
parser.add_argument(
    '-e', '--end', default=None, type=int,
    help='The frame number at which to stop. (Stop at the end by default.)')
args, _ = parser.parse_known_args()
debug = args.debug_out is not None


# Move expensive imports below argument parsing so that `--help` still runs quickly
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
from .track import track  # noqa: E402


videos = []
for ext in EXTS:
    videos.extend(glob.glob(os.path.join(args.path, f'*.{ext}')))

with open(args.out, 'w') as f:
    for video in tqdm.tqdm(videos):
        print(f'Processing {video}')
        filter_counts, debug_data = track(video, debug, args.start, args.end)
        filter_counts_array = np.array(filter_counts)
        mean, std = filter_counts_array.mean(), filter_counts_array.std()
        lower, upper = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        f.write(f'{video}: {np.round(mean):.0f}, [{max(lower, 0):.1f}, {upper:.1f}]\n')

        if debug:
            debug_path = os.path.join(args.debug_out, f'{os.path.basename(video)}.json')
            print(f'Writing debug output to {debug_path}')
            with open(debug_path, 'w') as dp:
                json.dump(debug_data, dp)
