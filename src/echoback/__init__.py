"""Echoback — offline voicemail transcription with a webhook callback."""

from importlib.metadata import PackageNotFoundError, version

try:
    # The single source of truth is the git tag: hatch-vcs derives the version at
    # build time, and the Docker build is handed it explicitly (the image has no
    # .git to read). Nothing in the tree carries a version number.
    __version__ = version("echoback")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
