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
        # Check if this specific file is already being logged to
        # Note: We check if ANY FileHandler is present, or specifically this one. 
        # For simplicity, if any FileHandler exists, we assume it's set up. 
        # But to be robust, we check if we are adding a new file.
        has_file_handler = False
        abs_log_file = os.path.abspath(log_file)
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if h.baseFilename == abs_log_file:
                    has_file_handler = True
                    break
        
        if not has_file_handler:
            # Ensure dir exists
            log_dir = os.path.dirname(abs_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(abs_log_file, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
