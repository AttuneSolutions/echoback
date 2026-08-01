"""Container entry point."""

from __future__ import annotations

import logging

import uvicorn

from .api import create_app
from .config import Config, load_config

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # Access logs would echo callback URLs; job-level logs carry what we need.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main() -> None:
    config: Config = load_config()
    configure_logging(config.log_level)
    uvicorn.run(
        create_app(config),
        host="0.0.0.0",  # noqa: S104 — container-internal; TLS terminates upstream
        port=config.port,
        log_level=config.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
