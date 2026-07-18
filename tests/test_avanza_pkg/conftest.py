"""Conftest for avanza package tests.

Ensures portfolio.avanza is importable before any test runs,
which is required for unittest.mock.patch() to resolve dotted
paths like "portfolio.avanza.auth._create_avanza_client" when
running under pytest-xdist (workers may not have imported the
subpackage yet).

2026-07-18: the unofficial `avanza` pip package is Windows/herc2-only —
it has never been installed on the Deck. An unguarded import here killed
WHOLE-suite collection on the Deck (pytest loads package conftests even
for --ignore'd dirs when tests/ is a package). When the dependency is
absent we skip collection of this entire directory instead.
"""

try:
    import portfolio.avanza  # noqa: F401 — force subpackage registration
    import portfolio.avanza.auth  # noqa: F401
    import portfolio.avanza.client  # noqa: F401

    _HAVE_AVANZA = True
except ImportError:
    _HAVE_AVANZA = False

collect_ignore_glob = [] if _HAVE_AVANZA else ["*"]
