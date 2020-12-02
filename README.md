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
