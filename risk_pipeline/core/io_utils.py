from __future__ import annotations

import io
import os
import sys
import tempfile
from typing import Union
from datetime import datetime
import logging


def write_atomic(path: str, data: Union[str, bytes]) -> None:
    """Atomically write data to path using a temporary file and os.replace.

    If data is str, it is encoded as UTF-8.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            if isinstance(data, str):
                f.write(data.encode("utf-8"))
            else:
                f.write(data)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class Tee:
    """Simple stdout/err tee to a file while keeping console output."""

    def __init__(self, filepath: str, stream):
        self.file = open(filepath, "a", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def tee_logger(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    sys.stdout = Tee(log_file, sys.stdout)
    sys.stderr = Tee(log_file, sys.stderr)


def setup_basic_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
