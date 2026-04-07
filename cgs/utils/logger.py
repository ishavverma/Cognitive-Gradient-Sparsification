"""Structured Logging."""

import logging
import sys


def get_logger(name: str = 'cgs', level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
