"""
Exception Theory Logging Utilities

Centralized logging configuration for Exception Theory library.

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
"""

import logging
import sys
from typing import Optional

# Default logger name
DEFAULT_LOGGER_NAME = 'ExceptionTheory'

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """
    Get the Exception Theory logger.
    
    Args:
        name: Logger name (default: 'ExceptionTheory')
    
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(DEFAULT_LOG_LEVEL)
        
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(DEFAULT_LOG_LEVEL)
            formatter = logging.Formatter(
                '[ET %(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
    
    return _logger


def set_log_level(level: int):
    """
    Set the global log level.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def enable_debug():
    """Enable debug logging."""
    set_log_level(logging.DEBUG)


def enable_info():
    """Enable info logging."""
    set_log_level(logging.INFO)


def enable_warning():
    """Enable warning logging."""
    set_log_level(logging.WARNING)


def disable_logging():
    """Disable all logging."""
    set_log_level(logging.CRITICAL + 1)


__all__ = [
    'get_logger',
    'set_log_level',
    'enable_debug',
    'enable_info',
    'enable_warning',
    'disable_logging',
    'DEFAULT_LOGGER_NAME',
]
