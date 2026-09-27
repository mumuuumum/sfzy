"""统一日志配置。"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """初始化根 logger，重复调用只生效一次。"""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root = logging.getLogger("sfzy")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    setup_logging(level)
    return logging.getLogger(f"sfzy.{name}" if name else "sfzy")
