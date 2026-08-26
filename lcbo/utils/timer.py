import logging
import time


class Timer:

    def __init__(self, logger: logging.Logger, name: str, log_level: int = logging.DEBUG):

        self.name = name
        self.logger = logger
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.log(self.log_level, f"Timer [{self.name}]: Elapsed time: {elapsed_time:.2f} seconds")
