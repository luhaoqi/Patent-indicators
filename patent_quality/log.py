import logging
import sys
import os
from typing import Optional


def get_logger(name: str = "patent_quality", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    # Add console handler if not present
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)

    # Add file handler if requested and not present
    if log_file:
        abs_log_file = os.path.abspath(log_file)
        # Remove existing FileHandler for the same file to ensure truncation
        to_remove = []
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_file:
                to_remove.append(h)
        for h in to_remove:
            try:
                logger.removeHandler(h)
                h.close()
            except Exception:
                pass
        # Ensure dir exists
        log_dir = os.path.dirname(abs_log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        # Open file in write mode to truncate each run
        fh = logging.FileHandler(abs_log_file, mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
