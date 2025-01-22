import functools
import io
import logging
import os
import traceback
from typing import Callable

from behavysis_pipeline.constants import CACHE_DIR

LOG_FILE_FORMAT = "%Y-%m-$d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def init_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
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
    logger.setLevel(level)
    # If logger does not have handlers, add them
    if not logger.hasHandlers():
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # File handler
        log_fp = os.path.join(CACHE_DIR, "debug.log")
        file_handler = logging.FileHandler(log_fp, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def init_logger_with_io_obj(name: str = __name__, level: int = logging.DEBUG) -> tuple[logging.Logger, io.StringIO]:
    # Making logger
    logger = init_logger(name, level)
    # Getting io_obj
    io_obj = None
    # Getting the io obj attached to one of the handlers
    for handler in logger.handlers:
        if isinstance(handler.stream, io.StringIO):  # type: ignore
            io_obj = handler.stream  # type: ignore
    # Adding io object to logger no io object handler is found
    if io_obj is None:
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        # Making io object
        io_obj = io.StringIO()
        # StringIO object handler
        io_handler = logging.StreamHandler(io_obj)
        io_handler.setLevel(level)
        io_handler.setFormatter(formatter)
        logger.addHandler(io_handler)
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
    print("MY LINE IS: ", log_line)
    datetime, name, level, message = log_line.split(" - ", 3)
    return datetime, name, level, message


def io_obj_to_msg(io_obj: io.StringIO) -> str:
    """
    Converts the io object logger stream to a string message
    (can be multi-line).
    """
    cursor = io_obj.tell()
    io_obj.seek(0)
    msg = ""
    for line in io_obj.readlines():
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        datetime, name, level, message = split_log_line(line)
        msg += f"{level} - {message}\n"
    io_obj.seek(cursor)
    return msg
