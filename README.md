# Multi-object tracking

To use this to count objects in a video:
```bash
git clone git@github.com:nuchi/multi_object_tracking.git
cd multi_object_tracking
pip install -e .  # installs in editable mode
mkdir output  # create a directory for debug output
python -m multi_object_tracking \
  --path $PATH_TO_VIDEOS_DIRECTORY \
  --out output.txt \
  --debug-out output
```

See also the other command-line options with
```bash
python -m multi_object_tracking --help
```

The `--debug-output $OUTPUT_DIRECTORY` option writes json-formatted debug output to a directory, which can be used to construct visualizations. See the notebook `debug.ipynb` which can be used to display the visualizations.

## Evaluating

To evaluate the performance against a labeled video, run
```bash
python -m multi_object_tracking.evaluate --input list_of_videos.txt --use-filters
```
where `list_of_videos.txt` is a file that looks like:
```
<path_to_video_1>,<path_to_directory_with_labels_for_video_1>
<path_to_video_2>,<path_to_directory_with_labels_for_video_2>
...
```
The labels for a video `my-video.avi` should have names like `my-video_0123_1_label.json` or `my-video_0123_1_labelbasic.json`.

## How it works

In broad strokes:

* Detect objects in a single frame ("detections"). (`observation.py`)
* Associate to each existing track (a constant velocity Kalman filter) a best detection (`association.py`)
* Update each track which received a detection (`update.py`)
* Create new tracks for detections that didn't correspond to an existing track (`update.py`)
* Delete tracks whenever they too closely mirror a "better" track (i.e. one with more certainty) (`update.py`)
* Delete tracks that haven't been observed recently (`track.py`)

The whole business is orchestrated in `track.py`, which is called via `__main__.py` or `evaluate.py`. See `debug.ipynb` for how to use the `visualize.py` module. Model parameters are stored in `default_params.json` and can be overridden with another file via `python -m multi_object_tracking[.evaluate] --params <path_to_alternate_params>` or by calling `params.init('my_alternate_params.json')`.

### To-do

* Improve the association phase by using `scipy.optimize.linear_sum_assignment` and minimizing total negative-log-likelihood.
