"""Crash-safe file output for the PRISM scripts.

Same behavior as ``benchmark/lib/json_io.py``; prism_monitor runs standalone from its own
directory (modules are imported as ``config`` / ``model``), so it carries its own copy.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output(path):
    """Yield a temporary path next to ``path``; on success it is synced to disk and replaces ``path``.

    ``os.replace`` makes the swap atomic: an interrupted run leaves either the old file or the
    new one, never a truncated file. On an exception the temporary file is removed and ``path``
    is left untouched.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        # mkstemp creates 0600 files, and writers such as safetensors' save_file recreate the file
        # with 0600 too; set normal permissions once the content is written
        os.chmod(tmp_path, 0o644)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path, obj, **dump_kwargs) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (see ``atomic_output``)."""
    with atomic_output(path) as tmp_path, open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **dump_kwargs)
