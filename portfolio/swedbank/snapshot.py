"""Pure snapshot derivation — no I/O, no network, no clock.

The broker export gives price, percent-since-purchase and market value per line,
but no quantity column. Both missing fields are recoverable:

    qty        = value_local / (price * fx)      # fx = 1 for locally-quoted lines
    cost_basis = value_local / (1 + pct/100)

The FX rate is not assumed, it is SOLVED: the correct rate is the one that drives
every foreign-quoted line to an integer share count simultaneously. A wrong rate
scatters them off-integer, so agreement across many lines at once is strong
evidence. Reconciling the reconstructed cost basis against the broker's stated
total is the independent second check.

Everything here is deterministic and side-effect free so it can be exhaustively
tested on synthetic books. Never seed a test from the live book — this repo is
public.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# Broker exports use non-breaking/thin spaces as thousands separators, a comma
# decimal mark, and U+2212 MINUS SIGN rather than ASCII hyphen. Getting any of
# these wrong yields a silently mis-parsed number rather than an error.
_SPACES = "     "
_MINUSES = "−–—"
_QTY_TOL = 1e-3
_MAX_SHARES = 100_000


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class RawRow:
    name: str
    price: float
    currency: str
    value_local: float
    since_purch_pct: float | None = None


@dataclass(frozen=True)
class Holding:
    key: str
    name: str
    qty: int
    currency: str
    cost_basis_local: float | None
    price_at_snapshot: float
    value_at_snapshot: float


def parse_number(raw: str) -> float:
    """Parse a broker-formatted number. Tolerates Swedish grouping and U+2212."""
    if raw is None:
        raise SnapshotError("empty number")
    s = str(raw).strip()
    for ch in _SPACES:
        s = s.replace(ch, "")
    for ch in _MINUSES:
        s = s.replace(ch, "-")
    s = s.replace("%", "").replace("+", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    if not s or s in {"-", "—"}:
        raise SnapshotError(f"unparseable number {raw!r}")
    try:
        return float(s)
    except ValueError as exc:
        raise SnapshotError(f"unparseable number {raw!r}") from exc


def _is_int(x: float, tol: float = _QTY_TOL) -> bool:
    return abs(x - round(x)) <= tol


def solve_fx(rows, reference_fx, base_currency="SEK", band=0.10, tol=_QTY_TOL):
    """Find the exact FX rate the broker used, given a live reference rate.

    The system is UNDERDETERMINED without a prior: for a single foreign row,
    5 shares at rate 8.5 is indistinguishable from 1 share at rate 42.5. Many
    varied rows over-constrain it so that only one rate survives, but relying on
    that is luck — a two-position account would resolve ambiguously and silently
    produce wrong quantities.

    So we use the prior we actually have. `reference_fx` is the live market rate
    (`portfolio.fx_rates.fetch_usd_sek()`); the broker's rate sits within a few
    percent of it. Candidates are generated from every foreign row, restricted to
    `reference_fx * (1 +/- band)`, then scored against all rows. Ties break toward
    the candidate closest to the reference.

    Returns (fx, diagnostics). Raises SnapshotError when no rate in the band
    explains every row.
    """
    foreign = [r for r in rows if r.currency != base_currency]
    if not foreign:
        return 1.0, {"foreign_rows": 0, "unanimous": True, "reference_fx": reference_fx}

    if not reference_fx or reference_fx <= 0:
        raise SnapshotError(
            "solve_fx requires a positive reference rate; without a prior the "
            "share-count solve is underdetermined and would silently pick a "
            "wrong rate"
        )

    lo, hi = reference_fx * (1 - band), reference_fx * (1 + band)
    for r in foreign:
        if r.price <= 0 or r.value_local <= 0:
            raise SnapshotError(f"row {r.name!r} has non-positive price/value")

    candidates = set()
    for r in foreign:
        n_lo = max(1, int(r.value_local / (r.price * hi)))
        n_hi = int(r.value_local / (r.price * lo)) + 1
        if n_hi - n_lo > _MAX_SHARES:
            continue
        for n in range(n_lo, n_hi + 1):
            fx = r.value_local / (r.price * n)
            if lo <= fx <= hi:
                candidates.add(round(fx, 10))

    if not candidates:
        raise SnapshotError(
            f"no FX candidate within {band:.0%} of reference {reference_fx:.4f} "
            f"produces integer share counts"
        )

    best = None
    for fx in candidates:
        hits, err = 0, 0.0
        for r in foreign:
            q = r.value_local / (r.price * fx)
            d = abs(q - round(q))
            if d <= tol:
                hits += 1
            err += d
        score = (hits, -err, -abs(fx - reference_fx))
        if best is None or score > best[0]:
            best = (score, fx, hits, err)

    _, fx, hits, err = best
    if hits != len(foreign):
        raise SnapshotError(
            f"no single FX rate within {band:.0%} of reference {reference_fx:.4f} "
            f"explains the export: best rate {fx:.6f} gives integer share counts "
            f"for {hits}/{len(foreign)} foreign-quoted rows (off-integer error "
            f"{err:.4f}). Refusing to guess — a wrong rate silently produces "
            f"wrong quantities."
        )
    return fx, {
        "foreign_rows": len(foreign),
        "integer_hits": hits,
        "total_error": err,
        "unanimous": True,
        "reference_fx": reference_fx,
        "deviation_from_reference": fx / reference_fx - 1.0,
        "candidates_considered": len(candidates),
    }


def derive_holdings(rows, fx, key_for, base_currency="SEK"):
    """Turn raw export rows into holdings with integer qty and cost basis."""
    out = []
    for r in rows:
        rate = fx if r.currency != base_currency else 1.0
        if r.price <= 0:
            raise SnapshotError(f"row {r.name!r} has non-positive price {r.price}")
        raw_qty = r.value_local / (r.price * rate)
        if not _is_int(raw_qty):
            raise SnapshotError(
                f"row {r.name!r} does not resolve to an integer share count "
                f"({raw_qty:.6f} at fx={rate:.6f})"
            )
        cost = None
        if r.since_purch_pct is not None:
            denom = 1 + r.since_purch_pct / 100.0
            if denom <= 0:
                raise SnapshotError(
                    f"row {r.name!r} since-purchase {r.since_purch_pct}% implies "
                    f"a non-positive cost basis"
                )
            cost = r.value_local / denom
        out.append(
            Holding(
                key=key_for(r.name),
                name=r.name,
                qty=int(round(raw_qty)),
                currency=r.currency,
                cost_basis_local=cost,
                price_at_snapshot=r.price,
                value_at_snapshot=r.value_local,
            )
        )
    return out


def reconcile(holdings, stated_value=None, stated_cost=None, cash=0.0, tol=1.0):
    """Check the reconstruction against the broker's own stated totals."""
    derived_value = sum(h.value_at_snapshot for h in holdings) + cash
    derived_cost = sum(h.cost_basis_local for h in holdings if h.cost_basis_local)
    report = {
        "derived_value": derived_value,
        "derived_cost": derived_cost,
        "value_ok": None,
        "cost_ok": None,
    }
    if stated_value is not None:
        report["value_delta"] = derived_value - stated_value
        report["value_ok"] = abs(report["value_delta"]) <= tol
    if stated_cost is not None:
        report["cost_delta"] = derived_cost - stated_cost
        # Cost basis is reconstructed from rounded percentages, so it accumulates
        # more error than the value column. Scale tolerance with position count.
        report["cost_ok"] = abs(report["cost_delta"]) <= max(tol, 0.5 * len(holdings))
    return report


def diff_holdings(old, new):
    """Compare two holding sets. Returns opened / closed / changed / unchanged."""
    o = {h.key: h for h in old}
    n = {h.key: h for h in new}
    opened = [n[k] for k in sorted(set(n) - set(o))]
    closed = [o[k] for k in sorted(set(o) - set(n))]
    changed, unchanged = [], []
    for k in sorted(set(o) & set(n)):
        if o[k].qty != n[k].qty:
            changed.append((o[k], n[k]))
        else:
            unchanged.append(n[k])
    return {
        "opened": opened,
        "closed": closed,
        "changed": changed,
        "unchanged": unchanged,
    }


_ROW_RE = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")


def parse_markdown_table(text, currency_hint="SEK"):
    """Parse a pasted broker table into RawRows.

    Expected columns: Holding | Last | Today % | Since purch. % | Value.
    Rows whose first cell is bold (a subtotal like **Equities**) are skipped,
    as are separator and header rows.
    """
    rows = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group("cells").split("|")]
        if len(cells) < 5:
            continue
        name = cells[0]
        if not name or name.startswith("**") or set(name) <= set("-: "):
            continue
        if name.lower() in {"holding", "instrument", "namn"}:
            continue
        price_cell = cells[1]
        if not price_cell or price_cell == "—":
            continue
        cur = currency_hint
        pc = price_cell
        for token in ("USD", "EUR", "NOK", "DKK", "SEK"):
            if token in pc:
                cur = token
                pc = pc.replace(token, "")
                break
        try:
            price = parse_number(pc)
            value = parse_number(cells[4])
        except SnapshotError:
            continue
        pct = None
        try:
            pct = parse_number(cells[3])
        except SnapshotError:
            pct = None
        rows.append(
            RawRow(
                name=name,
                price=price,
                currency=cur,
                value_local=value,
                since_purch_pct=pct,
            )
        )
    return rows


def rescale(holding, factor):
    return replace(holding, qty=int(round(holding.qty * factor)))
