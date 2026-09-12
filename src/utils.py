import sys
import json
from pathlib import Path
from loguru import logger
import numpy as np

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def setup_logging(log_file):
    """
    Configure loguru logging with both file and console output.
    
    Args:
        log_file: Path to the log file
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler with DEBUG level
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        compression="zip"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
