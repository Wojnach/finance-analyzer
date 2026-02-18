"""Shared pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require live GPU / network (deselect with '-k not integration')"
    )
