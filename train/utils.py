import logging
import inspect


def get_logger() -> logging.Logger:
    caller = inspect.stack()[1][3]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(caller)
