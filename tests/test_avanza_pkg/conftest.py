"""Conftest for avanza package tests.

Ensures portfolio.avanza is importable before any test runs,
which is required for unittest.mock.patch() to resolve dotted
paths like "portfolio.avanza.auth._create_avanza_client" when
running under pytest-xdist (workers may not have imported the
subpackage yet).
"""

import portfolio.avanza  # noqa: F401 — force subpackage registration
import portfolio.avanza.auth  # noqa: F401
import portfolio.avanza.client  # noqa: F401
