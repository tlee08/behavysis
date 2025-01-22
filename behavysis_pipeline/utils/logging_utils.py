import functools
import io
import logging
import os
import traceback
from typing import Callable

from behavysis_pipeline.constants import CACHE_DIR

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_IO_OBJ_FORMAT = "%(levelname)s - %(message)s"


def init_logger(
    name: str = __name__, console_level: int = logging.DEBUG, file_level: int = logging.DEBUG
) -> logging.Logger:
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
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # File handler
        log_fp = os.path.join(CACHE_DIR, "debug.log")
        file_handler = logging.FileHandler(log_fp, mode="a")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def init_logger_with_io_obj(
    name: str = __name__,
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    io_obj_level: int = logging.INFO,
) -> tuple[logging.Logger, io.StringIO]:
    # Making logger
    logger = init_logger(name, console_level=console_level, file_level=file_level)
    # Getting io_obj
    io_obj = None
    # Getting the io obj attached to one of the handlers
    for handler in logger.handlers:
        if isinstance(handler.stream, io.StringIO):  # type: ignore
            io_obj = handler.stream  # type: ignore
    # Adding io object to logger no io object handler is found
    if io_obj is None:
        # Formatter
        formatter = logging.Formatter(LOG_IO_OBJ_FORMAT)
        # Making io object
        io_obj = io.StringIO()
        # StringIO object handler
        io_handler = logging.StreamHandler(io_obj)
        io_handler.setLevel(io_obj_level)
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


def get_io_obj_content(io_obj: io.StringIO) -> str:
    """
    Gets the content from the StringIO object.
    Also restores cursor position of the StringIO object.
    """
    cursor = io_obj.tell()
    io_obj.seek(0)
    msg = io_obj.read()
    io_obj.seek(cursor)
    return msg
