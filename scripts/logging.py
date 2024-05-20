# scripts/logging.py

import sys
from loguru import logger


class Logger:
    def __init__(self):
        self.logger = logger

    def start(self):
        self.logger.remove()

        # Configure Loguru to log to standard output and a file
        self.logger.add(
            sys.stdout,
            level="INFO",
            format="{time} {level} {message}",
            backtrace=True,
            diagnose=True,
        )
        self.logger.add(
            "app.log",
            level="DEBUG",
            format="{time} {level} {message}",
            rotation="10 MB",
        )

        # Example of adding contextual information
        self.logger = self.logger.bind(application="text-generation")

        return self.logger
