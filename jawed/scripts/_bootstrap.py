"""Put the repo root on sys.path so scripts run without `pip install -e .`.

Import this first in each script::

    import _bootstrap  # noqa: F401
"""
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
