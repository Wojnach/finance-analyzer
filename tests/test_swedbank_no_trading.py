"""Structural guarantee: this subsystem cannot place an order.

The operator executes every trade by hand. A monitoring package that acquires
the ability to trade — even accidentally, via a transitive import — is the worst
failure this design can have. Enforce it mechanically rather than by convention,
so a future edit that imports an order module fails CI instead of shipping.
"""

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "portfolio" / "swedbank"

FORBIDDEN_MODULES = {
    "portfolio.avanza_orders",
    "portfolio.trade_guards",
    "portfolio.portfolio_mgr",
    "portfolio.golddigger",
    "portfolio.elongir",
    "portfolio.grid_fisher",
    "portfolio.fin_snipe",
    "portfolio.iskbets",
}

FORBIDDEN_CALLS = {
    "place_buy_order",
    "place_sell_order",
    "cancel_order",
    "_place_order",
}


def _source_files():
    return sorted(PKG.rglob("*.py"))


def test_package_exists():
    assert PKG.is_dir(), f"{PKG} missing"
    assert _source_files(), "no source files found"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_order_module_imported(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    bad = {m for m in imported if m in FORBIDDEN_MODULES}
    assert not bad, f"{path.name} imports order-capable module(s): {sorted(bad)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_order_function_referenced(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    bad = names & FORBIDDEN_CALLS
    assert not bad, f"{path.name} references order function(s): {sorted(bad)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_mutating_avanza_verbs(path):
    """api_post/api_delete against Avanza can mutate. Reads only."""
    src = path.read_text()
    for verb in ("api_delete(", "api_post("):
        assert verb not in src, (
            f"{path.name} uses {verb} — this package must issue read-only "
            f"Avanza calls (api_get) so it can never mutate account state"
        )
