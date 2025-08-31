"""
Logging utilities for the Pele processing system.
"""
import logging
import logging.handlers
from typing import Optional, Dict, Any
from pathlib import Path
import os
import sys
from datetime import datetime

from ..core.interfaces import Logger as LoggerInterface
from ..core.exceptions import FileSystemError


class PeleLogger(LoggerInterface):
    """Professional logging implementation."""

    def __init__(self, name: str = "pele_processing",
                 level: str = "INFO",
                 log_file: Optional[Path] = None,
                 console_output: bool = True,
                 include_rank: bool = False):

        self.name = name
        self.include_rank = include_rank
        self._setup_logger(level, log_file, console_output)

    def _setup_logger(self, level: str, log_file: Optional[Path], console_output: bool):
        """Setup logger with file and console handlers."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = self._create_formatter()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10 * 1024 * 1024, backupCount=5
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                raise FileSystemError("create", str(log_file), str(e))

    def _create_formatter(self) -> logging.Formatter:
        """Create log formatter."""
        format_parts = ['%(asctime)s']

        if self.include_rank:
            format_parts.append('[Rank %(rank)s]')

        format_parts.extend(['%(levelname)s', '%(name)s', '%(message)s'])
        format_str = ' - '.join(format_parts)

        return logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')

    def log_info(self, message: str, **kwargs) -> None:
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        self.logger.error(message, extra=kwargs)

    def log_debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, extra=kwargs)


class MPILogger(PeleLogger):
    """MPI-aware logger with per-rank files."""

    def __init__(self, name: str = "pele_processing",
                 level: str = "INFO",
                 log_directory: Optional[Path] = None,
                 console_output: bool = True):

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        except ImportError:
            self.rank = 0
            self.size = 1

        # Create rank-specific log file
        log_file = None
        if log_directory:
            log_directory = Path(log_directory)
            log_file = log_directory / f"{name}_rank_{self.rank}.log"

        super().__init__(name, level, log_file, console_output, include_rank=True)

    def log_info(self, message: str, **kwargs) -> None:
        kwargs['rank'] = self.rank
        super().log_info(message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        kwargs['rank'] = self.rank
        super().log_warning(message, **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        kwargs['rank'] = self.rank
        super().log_error(message, **kwargs)

    def log_debug(self, message: str, **kwargs) -> None:
        kwargs['rank'] = self.rank
        super().log_debug(message, **kwargs)


class ProgressLogger:
    """Progress tracking logger."""

    def __init__(self, logger: LoggerInterface, total_items: int,
                 report_interval: int = 10):
        self.logger = logger
        self.total_items = total_items
        self.report_interval = report_interval
        self.completed_items = 0
        self.start_time = datetime.now()

    def log_progress(self, item_name: str = "") -> None:
        """Log progress for completed item."""
        self.completed_items += 1

        if self.completed_items % self.report_interval == 0 or self.completed_items == self.total_items:
            percentage = (self.completed_items / self.total_items) * 100
            elapsed = datetime.now() - self.start_time

            if self.completed_items > 0:
                avg_time_per_item = elapsed.total_seconds() / self.completed_items
                remaining_items = self.total_items - self.completed_items
                eta_seconds = avg_time_per_item * remaining_items
                eta = datetime.now().replace(microsecond=0) + \
                      datetime.timedelta(seconds=int(eta_seconds))

                message = f"Progress: {self.completed_items}/{self.total_items} ({percentage:.1f}%) - ETA: {eta.strftime('%H:%M:%S')}"
            else:
                message = f"Progress: {self.completed_items}/{self.total_items} ({percentage:.1f}%)"

            if item_name:
                message += f" - Current: {item_name}"

            self.logger.log_info(message)


def create_logger(config_dict: Dict[str, Any]) -> LoggerInterface:
    """Factory function to create logger from configuration."""
    logger_type = config_dict.get('type', 'standard')

    if logger_type == 'mpi':
        return MPILogger(
            name=config_dict.get('name', 'pele_processing'),
            level=config_dict.get('level', 'INFO'),
            log_directory=Path(config_dict['log_directory']) if 'log_directory' in config_dict else None,
            console_output=config_dict.get('console_output', True)
        )
    else:
        return PeleLogger(
            name=config_dict.get('name', 'pele_processing'),
            level=config_dict.get('level', 'INFO'),
            log_file=Path(config_dict['log_file']) if 'log_file' in config_dict else None,
            console_output=config_dict.get('console_output', True)
        )


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[Path] = None,
                  use_mpi: bool = False) -> LoggerInterface:
    """Quick setup function for logging."""
    if use_mpi:
        return MPILogger(level=log_level, log_directory=log_file.parent if log_file else None)
    else:
        return PeleLogger(level=log_level, log_file=log_file)