"""Shared library for the finance-portfolio projects."""

import os


def offline() -> bool:
    """True when providers should read fixtures instead of the network."""
    return os.getenv("FINLIB_OFFLINE", "0") == "1"
