from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logger(level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger for the Worker and related components.

    This function is side-effectful by design but isolated in a small utility.
    """
    logger = logging.getLogger("eikon")
    if logger.handlers:
        # Already configured
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "eikon.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger