"""
Centralized logging configuration for the application.

Provides consistent logging across all modules with both console and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Import centralized data paths
# Note: This creates a circular import risk, so we import at function level if needed
# or rely on LOGS_DIR being set up first by data_paths module


def get_logs_dir():
    """Get logs directory, using data_paths if available."""
    try:
        from app.data_paths import LOGS_DIR

        return LOGS_DIR
    except ImportError:
        # Fallback for tests or standalone usage
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir


# Log file paths (evaluated lazily)
def get_log_paths():
    logs_dir = get_logs_dir()
    return logs_dir / "app.log", logs_dir / "error.log"


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

    # Get log paths
    APP_LOG, ERROR_LOG = get_log_paths()

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
    """Configure uvicorn loggers to match our format and ensure access logging."""
    import logging

    # Get uvicorn loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_error = logging.getLogger("uvicorn.error")

    # Ensure loggers have proper level
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_access.setLevel(logging.INFO)
    uvicorn_error.setLevel(logging.INFO)

    # Create console handler if none exists
    console_format = logging.Formatter(fmt='[%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')

    for logger in [uvicorn_logger, uvicorn_access, uvicorn_error]:
        # Check if handler exists
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            handler.setFormatter(console_format)
            logger.addHandler(handler)
        else:
            # Update existing handler formatting
            for handler in logger.handlers:
                handler.setFormatter(console_format)
