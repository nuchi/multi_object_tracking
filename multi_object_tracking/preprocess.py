from collections import deque
import itertools
import os

import cv2 as cv
import numpy as np

from . import params


def stream_video(filename):
    """
    Frames are rgb but with identical values; just read as greyscale.
    Takes a path and returns a generator consisting of greyscale frames.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Can't find file {filename}")

    vc = cv.VideoCapture(filename)
    success, frame = vc.read()
    if not success:
        raise ValueError(f'No frames found in {filename}, is it a valid video?')

    while True:
        yield frame[:, :, 0]
        success, frame = vc.read()
        if not success:
            break


def foreground(stream):
    """
    Takes a stream of greyscale images and yields a foreground segmentation.
    It assumes that the foreground consists of dark moving objects on a static
    background. It's okay if the background is dark in places, but returns won't
    be very good there. This is robust to overall changes in brightness.

    It works by taking the background to be the max of a moving window. The
    foreground is then simply the difference between the windowed maximum and the
    current frame.

    We could use just a single window (or the max across the entire video!) if we
    didn't need to handle changes in brightness, but to handle lightening over time
    we need to use a window to the past of the current frame. To handle darkening
    over time we need a window to the future of the current frame. Taking the min
    of these two windows produces a foreground that works in either case.
    """

    def left_window_max(stream, buflen):
        left_window = deque(itertools.islice(stream, 0, buflen), buflen)
        left_max = np.max(left_window, axis=0)
        for frame in left_window:
            yield left_max
        for frame in stream:
            left_window.append(frame)
            np.max(left_window, axis=0, out=left_max)
            yield left_max

    def right_window_max(stream, buflen):
        right_window = deque(itertools.islice(stream, 0, buflen), buflen)
        right_max = np.empty_like(right_window[0])
        for frame in stream:
            np.max(right_window, axis=0, out=right_max)
            yield right_max
            right_window.append(frame)
        np.max(right_window, axis=0, out=right_max)
        for frame in right_window:
            yield right_max

    buflen = params.PP_NUM_FRAMES_IN_MAX_BUFFER
    s1, s2, s3 = itertools.tee(stream, 3)
    for left_max, right_max, frame in zip(left_window_max(s1, buflen),
                                          right_window_max(s2, buflen),
                                          s3):
        yield np.minimum(left_max - frame, right_max - frame)
