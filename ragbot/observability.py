"""
Structured JSON logging.

One JSON object per line so the output can be grepped, shipped, or replayed
without a parser. Retrieval events carry the citations that were served, which
is what makes a bad answer diagnosable after the fact rather than only live.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

_LOGGER_NAME = "ragbot"
_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: str = None) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel((level or os.getenv("RAGBOT_LOG_LEVEL", "INFO")).upper())
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    configure()
    logging.getLogger(_LOGGER_NAME).log(
        level, event, extra={"event": event, "fields": fields}
    )
