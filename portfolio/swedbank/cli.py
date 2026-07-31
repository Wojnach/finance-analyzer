"""CLI: python -m portfolio.swedbank {show,sync,quotes}

`sync` is idempotent — re-running against the same export is a no-op. It always
prints a diff and requires confirmation before writing, so an unexpected change
(a mis-parsed export, a wrong FX solve) is visible before it lands.
"""

from __future__ import annotations

import argparse
import sys

from portfolio.swedbank import book as bookmod
from portfolio.swedbank.instruments import INSTRUMENTS
from portfolio.swedbank.pricing import sweep


def _fmt(x, dp=2):
    if x is None:
        return "-"
    return f"{x:,.{dp}f}".replace(",", " ")


def _fx_fn():
    from portfolio.fx_rates import fetch_usd_sek

    return fetch_usd_sek


def cmd_show(args):
    b = bookmod.load(args.path)
    s = sweep(keys=b.keys_held, fx_fn=_fx_fn())
    val = bookmod.revalue(b, s)

    for label, acc in val["accounts"].items():
        print(f"\n=== {label} ===")
        print(
            f"{'instrument':<28s}{'qty':>6s}{'mark':>11s}{'value':>14s}"
            f"{'P&L':>13s}{'P&L%':>8s}  flags"
        )
        for r in acc["holdings"]:
            flags = []
            if r["stale_last"]:
                flags.append("mid")
            if r["degraded"]:
                flags.append("degraded")
            if r["spread_pct"] and r["spread_pct"] > 0.5:
                flags.append(f"spread {r['spread_pct']:.1f}%")
            print(
                f"{r['name'][:27]:<28s}{r['qty']:>6d}{_fmt(r['mark']):>11s}"
                f"{_fmt(r['value']):>14s}{_fmt(r['pnl']):>13s}"
                f"{_fmt(r['pnl_pct'], 1):>8s}  {' '.join(flags)}"
            )
        print(
            f"{'':<28s}{'':>6s}{'cash':>11s}{_fmt(acc['cash']):>14s}\n"
            f"{'TOTAL':<28s}{'':>6s}{'':>11s}{_fmt(acc['total_value']):>14s}"
            f"{_fmt(acc['pnl']):>13s}{_fmt(acc['pnl_pct'], 1):>8s}"
        )

    t = val["total"]
    print("\n=== ALL ACCOUNTS ===")
    print(f"  value      {_fmt(t['total_value']):>16s} {val['base_currency']}")
    print(f"  cost       {_fmt(t['cost_basis']):>16s}")
    print(f"  unrealized {_fmt(t['pnl']):>16s}  ({_fmt(t['pnl_pct'], 2)}%)")
    print(f"  fx         {val['fx']}")
    print(f"  swept in   {val['sweep_duration_s']:.2f}s")

    if val["stale_last"]:
        print(f"\n  marked at mid (stale last): {[k for _, k in val['stale_last']]}")
    if val["degraded"]:
        print(f"  DEGRADED sources: {val['degraded']}")
    if val["unpriced"]:
        print(f"  UNPRICED (excluded from totals): {val['unpriced']}")
    return 0


def cmd_quotes(args):
    s = sweep(fx_fn=_fx_fn())
    print(f"{len(s.quotes)}/{len(INSTRUMENTS)} in {s.duration_s:.2f}s  fx={s.fx}")
    for k, q in s.quotes.items():
        flags = " ".join(
            f
            for f in (
                "STALE_LAST" if q.stale_last else "",
                "DEGRADED" if q.degraded else "",
            )
            if f
        )
        print(
            f"{k:<11s}{_fmt(q.mark):>11s} {q.mark_basis:<5s}"
            f"{('%.2f%%' % q.spread_pct) if q.spread_pct is not None else '-':>8s}"
            f"{(('%.0fs' % q.age_s) if q.age_s is not None else '-'):>7s}  "
            f"{q.source:<16s}{flags}"
        )
    for k, e in s.errors.items():
        print(f"  ! {k}: {e}")
    return 0 if s.ok else 1


def cmd_sync(args):
    from portfolio.swedbank.snapshot import (
        derive_holdings,
        diff_holdings,
        parse_markdown_table,
        solve_fx,
    )

    text = open(args.export, encoding="utf-8").read()
    rows = parse_markdown_table(text)
    if not rows:
        print("no parseable rows in export", file=sys.stderr)
        return 2

    ref = float(_fx_fn()())
    fx, diag = solve_fx(rows, reference_fx=ref)
    print(
        f"fx solved {fx:.6f} (reference {ref:.4f}, "
        f"{diag['deviation_from_reference'] * 100:+.3f}%), "
        f"{diag['integer_hits']}/{diag['foreign_rows']} integer share counts"
    )

    name_to_key = {i.name.lower(): k for k, i in INSTRUMENTS.items()}

    def key_for(name):
        n = name.strip().lower()
        if n in name_to_key:
            return name_to_key[n]
        # Collect ALL prefix candidates and refuse unless exactly one matches.
        # Returning the first silently mapped a broker-truncated name to the
        # wrong instrument — the Bitcoin and Ether tracker names share a long
        # prefix, so the holding would then be priced off the wrong orderbook.
        cands = sorted(
            {k for full, k in name_to_key.items()
             if full.startswith(n) or n.startswith(full)}
        )
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            raise KeyError(
                f"export row {name!r} is ambiguous — matches {cands}. Refusing to "
                f"guess; add an exact alias in portfolio/swedbank/instruments.py"
            )
        raise KeyError(
            f"cannot map export row {name!r} to a pinned instrument; "
            f"add it to portfolio/swedbank/instruments.py"
        )

    new = derive_holdings(rows, fx, key_for=key_for)
    print(f"derived {len(new)} positions")

    try:
        old_book = bookmod.load(args.path)
        old = [
            type(new[0])(
                key=p.key,
                name=p.key,
                qty=p.qty,
                currency=p.currency,
                cost_basis_local=p.cost_basis,
                price_at_snapshot=0.0,
                value_at_snapshot=0.0,
            )
            for a in old_book.accounts.values()
            for p in a.positions
        ]
    except bookmod.BookError:
        old = []

    d = diff_holdings(old, new)
    for h in d["opened"]:
        print(f"  + {h.key:<12s} {h.qty}")
    for h in d["closed"]:
        print(f"  - {h.key:<12s} CLOSED (was {h.qty})")
    for o, n in d["changed"]:
        print(f"  ~ {o.key:<12s} {o.qty} -> {n.qty}")
    if not (d["opened"] or d["closed"] or d["changed"]):
        print("  no changes")
        return 0

    if not args.yes:
        if input("write? [y/N] ").strip().lower() != "y":
            print("aborted")
            return 1
    # TODO: MANUAL REVIEW — writing is deliberately gated. A multi-account paste
    # carries no per-row account attribution, and guessing it would silently move
    # positions between accounts. Wire this once the export grows an account
    # column, or accept one export per account. Half-writing a book of real
    # positions is worse than not writing one.
    print(
        "\nsync is REPORT-ONLY: no write performed.\n"
        "  A multi-account export carries no per-row account attribution, so a "
        "write would have to\n  guess which account each position belongs to. "
        "Seed the book per-account instead."
    )
    # Exit non-zero: nothing was written, so reporting success would let a
    # caller believe a ledger now exists when the loop will still fail to load.
    return 3


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m portfolio.swedbank")
    p.add_argument("--path", default=bookmod.BOOK_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show").set_defaults(fn=cmd_show)
    sub.add_parser("quotes").set_defaults(fn=cmd_quotes)
    sy = sub.add_parser("sync")
    sy.add_argument("export")
    sy.add_argument("--yes", action="store_true")
    sy.set_defaults(fn=cmd_sync)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
