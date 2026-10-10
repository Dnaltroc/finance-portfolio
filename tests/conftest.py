import os

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """Every test runs on fixtures and a throwaway cache; no network, no keys."""
    monkeypatch.setenv("FINLIB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    yield
