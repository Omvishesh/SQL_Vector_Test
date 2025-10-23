#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-15

Centralized logging utilities with query ID management
"""

import logging
import threading
import contextvars
import os
from datetime import datetime
from typing import Optional, Union

# Context variables to store query IDs for the current request
query_id_context = contextvars.ContextVar('query_id', default=None)
main_query_id_context = contextvars.ContextVar('main_query_id', default=None)

# Worker id prefix: use process PID (safe when using uvicorn --workers)
WORKER_PREFIX = os.environ.get("WORKER_ID") or str(os.getpid())

class QueryIDManager:
    """Thread-safe query ID manager that resets daily and supports sub-queries"""

    def __init__(self):
        self._lock = threading.Lock()
        self._main_counter = 0
        self._sub_counters = {}  # {main_id_str: sub_counter}
        self._current_date = None
        self._reset_if_new_day()

    def _reset_if_new_day(self):
        """Reset counters if it's a new day"""
        today = datetime.now().strftime('%Y-%m-%d')
        if self._current_date != today:
            with self._lock:
                self._main_counter = 0
                self._sub_counters = {}
                self._current_date = today

    def get_next_main_id(self) -> str:
        """Get the next unique main query ID with worker prefix (pid.counter)"""
        self._reset_if_new_day()
        with self._lock:
            self._main_counter += 1
            return f"{WORKER_PREFIX}.{self._main_counter}"

    def get_next_sub_id(self, main_id: Union[int, str]) -> str:
        """Get the next unique sub-query ID for a given main query ID"""
        self._reset_if_new_day()
        main_key = str(main_id)
        with self._lock:
            if main_key not in self._sub_counters:
                self._sub_counters[main_key] = 0
            self._sub_counters[main_key] += 1
            return f"{main_key}.{self._sub_counters[main_key]}"

    def get_current_date(self) -> str:
        """Get the current date string"""
        return datetime.now().strftime('%Y-%m-%d')

# Global instance
query_id_manager = QueryIDManager()

class QueryIDFormatter(logging.Formatter):
    """Custom formatter that includes query ID in log messages"""

    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - [Query ID:%(query_id)s] - %(message)s'
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Get query ID from context (can be main ID or sub-query ID)
        query_id = query_id_context.get()
        if query_id is not None:
            record.query_id = str(query_id)
        else:
            record.query_id = "N/A"
        return super().format(record)

def setup_logging(filename_prefix: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging with query ID formatter

    Args:
        filename_prefix: Prefix for log filename (e.g., 'api', 'sql', 'metadata')
        level: Logging level

    Returns:
        Configured logger instance
    """
    current_date = query_id_manager.get_current_date()
    filename = f"{filename_prefix}-{current_date}.log"

    # Create formatter
    formatter = QueryIDFormatter()

    # Setup file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add only file handler (no console output)
    root_logger.addHandler(file_handler)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with query ID support"""
    return logging.getLogger(name)

def set_query_id(query_id: Union[int, str]) -> None:
    """Set the query ID for the current context"""
    query_id_context.set(query_id)

def get_query_id() -> Optional[Union[int, str]]:
    """Get the current query ID from context (can be main ID or sub-query ID)"""
    return query_id_context.get()

def get_main_query_id() -> Optional[Union[int, str]]:
    """Get the current main query ID from context"""
    return main_query_id_context.get()

def clear_query_id() -> None:
    """Clear the query ID from context"""
    query_id_context.set(None)
    main_query_id_context.set(None)

class QueryIDContext:
    """Context manager for main query ID"""

    def __init__(self, main_query_id: Union[int, str]):
        self.main_query_id = main_query_id
        self.query_token = None
        self.main_token = None

    def __enter__(self):
        self.query_token = query_id_context.set(self.main_query_id)
        self.main_token = main_query_id_context.set(self.main_query_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.query_token:
            query_id_context.reset(self.query_token)
        if self.main_token:
            main_query_id_context.reset(self.main_token)

class SubQueryIDContext:
    """Context manager for sub-query ID"""

    def __init__(self, sub_query_id: str, main_query_id: Union[int, str]):
        self.sub_query_id = sub_query_id
        self.main_query_id = main_query_id
        self.query_token = None
        self.main_token = None

    def __enter__(self):
        self.query_token = query_id_context.set(self.sub_query_id)
        self.main_token = main_query_id_context.set(self.main_query_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.query_token:
            query_id_context.reset(self.query_token)
        if self.main_token:
            main_query_id_context.reset(self.main_token)