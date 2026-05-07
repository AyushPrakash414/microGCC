"""
Centralized Logging Module.

Provides a consistent logging setup across the entire application
with both console and rotating file handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_yaml_config


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    cfg = get_yaml_config()
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(cfg.log_format)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    log_path = PROJECT_ROOT / cfg.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
