"""
Centralized logging configuration for the application.

Provides consistent logging across all modules with both console and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file paths
APP_LOG = LOGS_DIR / "app.log"
ERROR_LOG = LOGS_DIR / "error.log"


def setup_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler with color-coded output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(fmt='[%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)

    # File handler for all logs (rotating, max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(APP_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')  # 10MB
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Error handler for ERROR and CRITICAL only (rotating)
    error_handler = RotatingFileHandler(ERROR_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')  # 10MB
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return setup_logger(name)


# Convenience functions for direct logging
_default_logger = setup_logger("app")


def debug(message: str, **kwargs):
    """Log debug message."""
    _default_logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message."""
    _default_logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message."""
    _default_logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message."""
    _default_logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message."""
    _default_logger.critical(message, **kwargs)


def log_exception(message: str = "An exception occurred"):
    """Log an exception with full traceback."""
    _default_logger.exception(message)


# Configure uvicorn logging to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format."""
    # Get uvicorn loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access = logging.getLogger("uvicorn.access")

    # Set consistent formatting
    for handler in uvicorn_logger.handlers:
        handler.setFormatter(logging.Formatter(fmt='[%(levelname)s] uvicorn: %(message)s', datefmt='%H:%M:%S'))

    for handler in uvicorn_access.handlers:
        handler.setFormatter(logging.Formatter(fmt='[%(levelname)s] uvicorn.access: %(message)s', datefmt='%H:%M:%S'))
