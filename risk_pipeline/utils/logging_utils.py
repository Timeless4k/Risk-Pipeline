"""
Logging utilities for RiskPipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config.logging_config import get_logging_config, apply_logging_config


def setup_logging(log_file_path: Optional[str] = None, 
                 level: int = logging.INFO,
                 format_string: Optional[str] = None,
                 date_format: Optional[str] = None,
                 verbose: bool = False,
                 quiet: bool = False) -> logging.Logger:
    """
    Setup comprehensive logging configuration with reduced verbosity.
    
    Args:
        log_file_path: Path to log file. If None, creates timestamped file in logs directory.
        level: Logging level
        format_string: Custom format string for log messages
        date_format: Custom date format string
        verbose: Enable verbose logging (DEBUG level)
        quiet: Enable minimal logging (WARNING level only)
        
    Returns:
        Configured logger instance
    """
    # Get logging configuration
    if quiet:
        config = get_logging_config(verbose=False)
        config['root_level'] = logging.WARNING
        config['console_level'] = logging.WARNING
    elif verbose:
        config = get_logging_config(verbose=True)
        level = logging.DEBUG
    else:
        config = get_logging_config(verbose=False)
    
    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create logs directory (ensure file path, not dir, is passed below)
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Generate log file path if not provided
    if log_file_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = log_dir / f'pipeline_run_{timestamp}.log'
    else:
        log_file_path = Path(log_file_path)
        # If a directory path is provided, create a default log file inside it
        if log_file_path.is_dir() or str(log_file_path).endswith('/'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_path = log_file_path / f'pipeline_run_{timestamp}.log'
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    if format_string is None:
        format_string = config['format']
    
    if date_format is None:
        date_format = config['date_format']
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # File handler - captures logs to file with configured level
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(config['file_level'])
    file_handler.setFormatter(formatter)
    
    # Console handler - less verbose for console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config['console_level'])
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(config['root_level'])
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Apply component-specific logging configuration
    apply_logging_config(config)
    
    # Create pipeline-specific logger
    logger = logging.getLogger('RiskPipeline')
    logger.info(f"Logging initialized - File: {log_file_path}, Level: {logging.getLevelName(level)}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_exception(self, message: str):
        """Log exception with traceback."""
        self.logger.exception(message) 