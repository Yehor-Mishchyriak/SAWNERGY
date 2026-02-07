import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime

_logger = logging.getLogger(__name__)


def configure_logging(
    logs_dir: Path | str,
    file_level: int = logging.WARNING,
    console_level: int = logging.WARNING,
    force: bool = False,
) -> None:
    """
    Configure a logger with a timed rotating file handler and console handler.

    Args:
        logs_dir: Directory where log files will be stored.
        file_level: Logging level for the file handler (default: WARNING).
        console_level: Logging level for the console handler (default: WARNING).
        force: If True, remove existing handlers on the root logger before
            configuring new ones.
    """

    if isinstance(logs_dir, str):
        logs_dir = Path(logs_dir)

    root = logging.getLogger()
    if root.handlers:
        if not force:
            return
        _logger.info("configure_logging(force=True): clearing %d existing handler(s)", len(root.handlers))
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                _logger.debug("Failed to close handler %r during force reset", handler, exc_info=True)

    logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    logfile = logs_dir / f"sawnergy_{datetime.now():%Y-%m-%d_%H%M%S}.log"

    file_h = TimedRotatingFileHandler(
        logfile,
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    file_h.setLevel(file_level)
    file_h.setFormatter(formatter)

    console_h = logging.StreamHandler()
    console_h.setLevel(console_level)
    console_h.setFormatter(formatter)

    # ensure root level is low enough to handle both handlers
    root.setLevel(min(file_level, console_level))
    root.addHandler(file_h)
    root.addHandler(console_h)


__all__ = ["configure_logging"]
