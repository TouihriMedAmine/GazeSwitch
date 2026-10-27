"""Logging setup: console + rotating file."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO", file_path: str = "logs/gazeswitch.log") -> logging.Logger:
    logger = logging.getLogger("gazeswitch")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:  # already configured (e.g. tests calling twice)
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    if sys.stderr is not None:  # None in the windowless (pythonw / .exe) app
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    try:
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", file_path, exc)

    # Library modules log under their package name ("src.targeting", ...).
    package_logger = logging.getLogger("src")
    package_logger.setLevel(logger.level)
    package_logger.propagate = False
    for handler in logger.handlers:
        package_logger.addHandler(handler)

    # Quiet down chatty third-party loggers.
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    return logger
