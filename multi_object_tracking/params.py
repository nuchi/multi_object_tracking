import json
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import multi_object_tracking


def init(filename=None):
    with (open(filename) if filename is not None else
          pkg_resources.open_text(multi_object_tracking, 'default_params.json')) as f:
        params = json.load(f)
        globals().update(**params)


init()
