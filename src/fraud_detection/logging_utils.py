"""Project-wide logging helper — consistent, timestamped, single config point."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Return a configured logger. Idempotent across calls."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _CONFIGURED = True
    return logging.getLogger(name)
