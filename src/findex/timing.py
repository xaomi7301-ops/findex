"""Elapsed-time decorator: closure + functools.wraps preserve the function API."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger("findex.timing")


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Log a duration even when the wrapped function raises."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("@timed %s: %.3f ms", func.__name__, (time.perf_counter() - start) * 1000)

    return wrapped
