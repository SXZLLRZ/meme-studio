"""Keep provider credentials and private request details out of application logs."""

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


_PRIVATE_FIELD = re.compile(
    r"""(?ix)\b(authorization|(?:dashscope_)?api[_-]?key|base64)\b["']?\s*[:=]\s*"""
    r"""(?:"[^"]*"|'[^']*'|(?:Bearer\s+)?[^\s,;}]+)"""
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_URL = re.compile(r"""(?i)\bhttps?://[^\s"'<>]+""")
_DATA_URI = re.compile(r"""(?i)\bdata:[^\s"',]+;base64,[a-z0-9+/=_-]+""")
_BASE64_BLOB = re.compile(r"(?<![a-zA-Z0-9+/_-])[a-zA-Z0-9+/_-]{128,}={0,2}(?![a-zA-Z0-9+/_-])")


class RedactingFormatter(logging.Formatter):
    def __init__(self, secret: str = "") -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")
        self._secret = secret

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self._secret:
            message = message.replace(self._secret, "[REDACTED]")
        message = _PRIVATE_FIELD.sub(r"\1=[REDACTED]", message)
        message = _BEARER.sub("[REDACTED]", message)
        message = _URL.sub("[URL REDACTED]", message)
        message = _DATA_URI.sub("[DATA REDACTED]", message)
        return _BASE64_BLOB.sub("[DATA REDACTED]", message)


def configure_logging(directory: Path, api_key: str = "") -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("meme_studio")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = RotatingFileHandler(directory / "app.log", maxBytes=1024 * 1024, backupCount=2, encoding="utf-8")
    handler.setFormatter(RedactingFormatter(api_key))
    logger.addHandler(handler)
    return logger
