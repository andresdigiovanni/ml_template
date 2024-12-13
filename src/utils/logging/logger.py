import logging


class Logger:
    """
    Logger class for logging messages during the execution of the application.

    This class wraps Python's built-in logging module to provide a consistent
    interface for logging messages at various levels (INFO, DEBUG, WARNING, ERROR, CRITICAL).

    Attributes:
        logger (logging.Logger): The logger instance used for logging messages.
    """

    def __init__(
        self,
        name: str = "ApplicationLogger",
        log_level: int = logging.INFO,
        log_file: str = None,
    ):
        """
        Initializes the Logger instance.

        Args:
            name (str): Name of the logger.
            log_level (int): Logging level (default is logging.INFO).
            log_file (str, optional): Path to the log file. If provided, logs will be written to this file.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add console handler to the logger
        self._logger.addHandler(console_handler)

        if log_file:
            # Create file handler if log file is specified
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def debug(self, message: str):
        """Logs a message with DEBUG level."""
        self._logger.debug(message)

    def info(self, message: str):
        """Logs a message with INFO level."""
        self._logger.info(message)

    def warning(self, message: str):
        """Logs a message with WARNING level."""
        self._logger.warning(message)

    def error(self, message: str):
        """Logs a message with ERROR level."""
        self._logger.error(message)

    def critical(self, message: str):
        """Logs a message with CRITICAL level."""
        self._logger.critical(message)
