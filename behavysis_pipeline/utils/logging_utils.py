import functools
import io
import logging
import os
import traceback
from typing import Callable

from behavysis_pipeline.constants import CACHE_DIR

LOG_FILE_FORMAT = "%Y-%m-$d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def init_logger(name: str = __name__) -> logging.Logger:
    """
    Setup logging configuration.

    Logs to:
    - console
    - file (<cache_dir>/debug.log)
    """
    # Making cache directory if it does not exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Initialising/getting logger and its configuration
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # If logger does not have handlers, add them
    if not logger.hasHandlers():
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # File handler
        log_fp = os.path.join(CACHE_DIR, "debug.log")
        file_handler = logging.FileHandler(log_fp, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Returning logger
    return logger


def init_logger_with_io_obj(name: str = __name__) -> tuple[logging.Logger, io.StringIO]:
    # Making logger
    logger = init_logger(name)
    # Adding io object to logger
    if len(logger.handlers) == 2:
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        # StringIO object handler
        io_obj = io.StringIO()
        console_handler = logging.StreamHandler(io_obj)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # Returning logger
    return logger, io_obj


def logger_func_decorator(logger: logging.Logger):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"STARTED {func.__name__}")
                output = func(*args, **kwargs)
                logger.info(f"FINISHED {func.__name__}")
                return output
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                raise e

        return wrapper

    return decorator


def split_log_line(log_line: str) -> tuple[str, str, str, str]:
    """
    Splits the log line into the datetime, name, level, and message.
    """
    datetime, name, level, message = log_line.split(" - ", 3)
    return datetime, name, level, message
