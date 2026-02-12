"""Logging utilities for consistent formatting across the project.

This module provides a standardized logging setup for training, inference,
and evaluation scripts. All loggers use consistent formatting with timestamps,
log levels, and module names.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with consistent formatting.
    
    Creates a logger with both console and optional file output. The logger
    uses a standardized format showing timestamp, level, module name, and message.
    
    Args:
        name: Name of the logger (typically __name__ from calling module)
        log_file: Optional path to log file. If provided, creates parent
            directories if they don't exist. If None, only logs to console.
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Optional custom format string. If None, uses default format.
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__, "logs/training.log")
        >>> logger.info("Training started")
        2024-02-12 10:30:45 - INFO - __main__ - Training started
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatters
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.
    
    Retrieves a logger that was previously configured with setup_logger().
    If the logger doesn't exist or wasn't configured, returns a basic logger.
    
    Args:
        name: Name of the logger to retrieve
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Using existing logger")
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Change logging level for an existing logger.
    
    Updates the log level for both the logger and all its handlers.
    
    Args:
        logger: Logger instance to update
        level: New logging level (e.g., logging.DEBUG, logging.WARNING)
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
