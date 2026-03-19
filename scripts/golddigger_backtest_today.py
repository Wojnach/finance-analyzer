"""GoldDigger backtest on today's data — iterate parameters until we find
a configuration that detects real gold moves.

Uses 1-minute klines from Binance FAPI as proxy for 5s polling.
"""

import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portfolio.golddigger.augmented_signals import AugmentedSignals
from portfolio.golddigger.data_provider import MarketSnapshot
from portfolio.golddigger.signal import CompositeSignal
from portfolio.http_retry import fetch_with_retry

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
SESSION_START_H, SESSION_START_M = 15, 30
SESSION_END_H, SESSION_END_M = 21, 55


def fetch_klines_today(symbol="XAUUSDT", interval="1m", days_ago=0):
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        tz = timezone(timedelta(hours=1))
    now_cet = datetime.now(tz) - timedelta(days=days_ago)
    start_of_day = now_cet.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(start_of_day.astimezone(UTC).timestamp() * 1000)
    all_klines = []
    start = start_ms
    while True:
        r = fetch_with_retry(
            f"{BINANCE_FAPI}/klines",
            params={"symbol": symbol, "interval": interval, "startTime": start, "limit": 1500},
            timeout=15,
        )
        if r is None:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_klines.extend(batch)
        start = batch[-1][0] + 60000
        if len(batch) < 1500:
            break
    return all_klines


def fetch_usdsek_today():
    try:
        from portfolio.fx_rates import fetch_usd_sek
        rate = fetch_usd_sek()
        if rate:
            return rate
    except Exception:
        pass
    r = fetch_with_retry("https://api.frankfurter.app/latest?from=USD&to=SEK", timeout=10)
    if r and r.ok:
        data = r.json()
        return data.get("rates", {}).get("SEK", 10.5)
    return 10.5


def run_single(klines, usdsek, us10y, theta_in, theta_out, confirm_polls,
               w_gold, w_fx, w_yield, window_n, min_window, verbose=False):
    """Run one backtest pass and return alerts."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        tz = timezone(timedelta(hours=1))

    signal = CompositeSignal(
        window_n=window_n,
        min_window=min_window,
        w_gold=w_gold,
        w_fx=w_fx,
        w_yield=w_yield,
        theta_in=theta_in,
        theta_out=theta_out,
        confirm_polls=confirm_polls,
    )

    alerts = []
    all_states = []
    in_position = False
    entry_price = 0.0
    entry_time = ""

    for kline in klines:
        ts_ms = kline[0]
        close = float(kline[4])
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
        dt_cet = dt_utc.astimezone(tz)
        hour, minute = dt_cet.hour, dt_cet.minute
        time_str = dt_cet.strftime("%H:%M")
        now_mins = hour * 60 + minute
        in_session = (SESSION_START_H * 60 + SESSION_START_M) <= now_mins <= (SESSION_END_H * 60 + SESSION_END_M)

        snap = MarketSnapshot(ts_utc=dt_utc, gold=close, usdsek=usdsek, us10y=us10y)
        state = signal.update(snap)

        record = {
            "time": time_str, "gold": close, "S": state.composite_s,
            "z_gold": state.z_gold, "confirm": state.confirm_count,
            "valid": state.valid, "in_session": in_session,
        }
        all_states.append(record)

        if not state.valid:
            continue

        # Entry
        if in_session and not in_position and signal.should_enter(state):
            entry_price = close
            entry_time = time_str
            in_position = True
            alerts.append({
                "type": "ENTRY", "time": time_str, "gold": close,
                "S": state.composite_s, "z_gold": state.z_gold,
                "confirms": state.confirm_count,
            })

        # Exit
        elif in_position and signal.should_exit(state):
            pnl_pct = (close - entry_price) / entry_price * 100
            in_position = False
            alerts.append({
                "type": "EXIT", "time": time_str, "gold": close,
                "S": state.composite_s, "z_gold": state.z_gold,
                "pnl_pct": pnl_pct, "entry_price": entry_price,
                "entry_time": entry_time,
            })

    return alerts, all_states


def main():
    print("=" * 70)
    print("GOLDDIGGER PARAMETER SWEEP — Finding the right config")
    print("=" * 70)

    klines = fetch_klines_today()
    if not klines:
        print("ERROR: No klines fetched")
        return
    print(f"Fetched {len(klines)} klines")

    usdsek = fetch_usdsek_today()
    us10y = 0.04  # constant placeholder
    print(f"USD/SEK: {usdsek:.4f}")
    print()

    # Parameter grid to sweep
    configs = [
        # (label, theta_in, theta_out, confirms, w_gold, w_fx, w_yield, window_n, min_window)
        ("Original (too strict)",   1.0, 0.2, 6, 0.50, 0.30, 0.20, 720, 60),
        ("Lower confirms (3)",      1.0, 0.2, 3, 0.50, 0.30, 0.20, 720, 60),
        ("Lower confirms (2)",      1.0, 0.2, 2, 0.50, 0.30, 0.20, 720, 60),
        ("Lower theta_in (0.7)",    0.7, 0.1, 3, 0.50, 0.30, 0.20, 720, 60),
        ("Gold-only (w=1.0)",       0.7, 0.1, 3, 1.00, 0.00, 0.00, 720, 60),
        ("Gold-only lower (0.5)",   0.5, 0.0, 2, 1.00, 0.00, 0.00, 720, 60),
        ("Shorter window (120)",    0.7, 0.1, 3, 1.00, 0.00, 0.00, 120, 20),
        ("Short window + low thr",  0.5, 0.0, 2, 1.00, 0.00, 0.00, 120, 20),
        ("Ultra-responsive",        0.5, -0.2, 2, 1.00, 0.00, 0.00, 60, 15),
    ]

    best_config = None
    best_alerts = []
    best_label = ""

    for label, theta_in, theta_out, confirms, wg, wf, wy, wn, mw in configs:
        alerts, states = run_single(
            klines, usdsek, us10y,
            theta_in=theta_in, theta_out=theta_out, confirm_polls=confirms,
            w_gold=wg, w_fx=wf, w_yield=wy, window_n=wn, min_window=mw,
        )
        entries = [a for a in alerts if a["type"] == "ENTRY"]
        exits = [a for a in alerts if a["type"] == "EXIT"]
        session_states = [s for s in states if s["in_session"] and s["valid"]]
        s_range = ""
        if session_states:
            smin = min(s["S"] for s in session_states)
            smax = max(s["S"] for s in session_states)
            s_range = f"S range [{smin:.2f}, {smax:.2f}]"

        status = f"  {label:30s} | {len(entries)}E {len(exits)}X | theta={theta_in}/{theta_out} conf={confirms} w={wg}/{wf}/{wy} win={wn} | {s_range}"
        print(status)

        if alerts and (not best_alerts or len(alerts) > len(best_alerts)):
            best_config = (theta_in, theta_out, confirms, wg, wf, wy, wn, mw)
            best_alerts = alerts
            best_label = label

    print()
    if not best_alerts:
        print("No config produced alerts. Trying even more aggressive parameters...")
        print()

        # More aggressive sweep
        aggressive_configs = [
            ("Micro window (30)",       0.4, -0.3, 1, 1.00, 0.00, 0.00, 30, 10),
            ("Micro + confirm=2",       0.4, -0.3, 2, 1.00, 0.00, 0.00, 30, 10),
            ("Tight window (20/8)",      0.3, -0.2, 2, 1.00, 0.00, 0.00, 20, 8),
            ("Direction only (0.2)",    0.2, -0.2, 2, 1.00, 0.00, 0.00, 30, 10),
            ("1-sigma (theta=1.0) micro",1.0, 0.0, 1, 1.00, 0.00, 0.00, 30, 10),
        ]

        for label, theta_in, theta_out, confirms, wg, wf, wy, wn, mw in aggressive_configs:
            alerts, states = run_single(
                klines, usdsek, us10y,
                theta_in=theta_in, theta_out=theta_out, confirm_polls=confirms,
                w_gold=wg, w_fx=wf, w_yield=wy, window_n=wn, min_window=mw,
            )
            entries = [a for a in alerts if a["type"] == "ENTRY"]
            exits = [a for a in alerts if a["type"] == "EXIT"]
            session_states = [s for s in states if s["in_session"] and s["valid"]]
            s_range = ""
            if session_states:
                smin = min(s["S"] for s in session_states)
                smax = max(s["S"] for s in session_states)
                s_range = f"S range [{smin:.2f}, {smax:.2f}]"

            status = f"  {label:30s} | {len(entries)}E {len(exits)}X | theta={theta_in}/{theta_out} conf={confirms} win={wn} | {s_range}"
            print(status)

            if alerts and (not best_alerts or len(alerts) > len(best_alerts)):
                best_config = (theta_in, theta_out, confirms, wg, wf, wy, wn, mw)
                best_alerts = alerts
                best_label = label

    print()
    print("=" * 70)
    if best_alerts:
        print(f"BEST CONFIG: {best_label}")
        print(f"  theta_in={best_config[0]}, theta_out={best_config[1]}, confirms={best_config[2]}")
        print(f"  w_gold={best_config[3]}, w_fx={best_config[4]}, w_yield={best_config[5]}")
        print(f"  window={best_config[6]}, min_window={best_config[7]}")
        print()
        print("ALERTS:")
        for a in best_alerts:
            if a["type"] == "ENTRY":
                print(f"  >> ENTRY {a['time']} | Gold=${a['gold']:.1f} | S={a['S']:.3f} | z_gold={a['z_gold']:.3f} | confirms={a['confirms']}")
            else:
                print(f"  << EXIT  {a['time']} | Gold=${a['gold']:.1f} | S={a['S']:.3f} | P&L={a['pnl_pct']:+.3f}% (entry ${a['entry_price']:.1f} @ {a['entry_time']})")

        # Now run detailed timeline for best config
        print()
        print("DETAILED TIMELINE (best config, session hours, every 5 min):")
        _, states = run_single(
            klines, usdsek, us10y,
            theta_in=best_config[0], theta_out=best_config[1], confirm_polls=best_config[2],
            w_gold=best_config[3], w_fx=best_config[4], w_yield=best_config[5],
            window_n=best_config[6], min_window=best_config[7],
        )
        session_states = [s for s in states if s["in_session"] and s["valid"]]
        print(f"  {'Time':>5s}  {'S(t)':>7s}  {'z_gold':>7s}  {'Gold':>8s}  {'Conf':>4s}  Note")
        print(f"  {'-----':>5s}  {'------':>7s}  {'------':>7s}  {'------':>8s}  {'----':>4s}  ----")
        last_printed = ""
        for s in session_states:
            t = s["time"]
            mm = int(t.split(":")[1])
            if mm % 5 == 0 and t != last_printed:
                note = ""
                sv = s["S"]
                if sv >= best_config[0]:
                    note = " <<< ENTRY ZONE"
                    if s["confirm"] >= best_config[2]:
                        note = " *** ENTRY ALERT ***"
                elif sv <= best_config[1]:
                    note = " >>> EXIT ZONE"
                elif sv >= best_config[0] * 0.7:
                    note = " ~ building"
                print(f"  {t:>5s}  {sv:>7.3f}  {s['z_gold']:>7.3f}  ${s['gold']:>7.1f}  {s['confirm']:>4d}{note}")
                last_printed = t

        # Gold price range
        if session_states:
            gold_high = max(s["gold"] for s in session_states)
            gold_low = min(s["gold"] for s in session_states)
            gold_range_pct = (gold_high - gold_low) / gold_low * 100
            print(f"\n  Gold range: ${gold_low:.1f} - ${gold_high:.1f} ({gold_range_pct:.2f}%, cert={gold_range_pct*20:.1f}%)")
    else:
        print("NO CONFIG PRODUCED ALERTS — market was truly flat/choppy today")
        # Show raw gold moves for context
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo("Europe/Stockholm")
        except ImportError:
            tz = timezone(timedelta(hours=1))

        session_klines = []
        for kl in klines:
            dt = datetime.fromtimestamp(kl[0]/1000, tz=UTC).astimezone(tz)
            h, m = dt.hour, dt.minute
            nm = h*60+m
            if (SESSION_START_H*60+SESSION_START_M) <= nm <= (SESSION_END_H*60+SESSION_END_M):
                session_klines.append((dt.strftime("%H:%M"), float(kl[4])))
        if session_klines:
            prices = [p for _, p in session_klines]
            print(f"\n  Gold session: ${min(prices):.1f} - ${max(prices):.1f} ({(max(prices)-min(prices))/min(prices)*100:.2f}%)")
            # Show 15-min OHLC
            print("\n  15-min gold candles:")
            bucket = {}
            for t, p in session_klines:
                key = t[:4] + "0" if int(t[3:5]) < 15 else (t[:4] + "5" if int(t[3:5]) < 30 else (t[:3] + "30" if int(t[3:5]) < 45 else t[:3] + "45"))
                # simpler: just bucket by 15 min
                h = int(t[:2])
                m = int(t[3:5])
                bm = (m // 15) * 15
                bkey = f"{h:02d}:{bm:02d}"
                if bkey not in bucket:
                    bucket[bkey] = []
                bucket[bkey].append(p)
            for bk in sorted(bucket.keys()):
                ps = bucket[bk]
                o, c = ps[0], ps[-1]
                hi, lo = max(ps), min(ps)
                chg = (c - o) / o * 100
                bar = "+" * int(abs(chg) * 20) if chg > 0 else "-" * int(abs(chg) * 20)
                color_prefix = "^" if chg > 0 else "v" if chg < 0 else "="
                print(f"    {bk}  O={o:.1f} H={hi:.1f} L={lo:.1f} C={c:.1f}  {chg:+.3f}%  {color_prefix}{bar}")


def run_augmented_filter(klines, usdsek, us10y, config_tuple, label):
    """Re-run the best config with augmented signal filtering.

    Takes klines + best config tuple, computes augmented signals at each entry
    point, and shows which entries would have been filtered.
    """
    theta_in, theta_out, confirms, wg, wf, wy, wn, mw = config_tuple

    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        tz = timezone(timedelta(hours=1))

    # First, run the base signal to find entry points
    alerts, _ = run_single(
        klines, usdsek, us10y,
        theta_in=theta_in, theta_out=theta_out, confirm_polls=confirms,
        w_gold=wg, w_fx=wf, w_yield=wy, window_n=wn, min_window=mw,
    )
    entries = [a for a in alerts if a["type"] == "ENTRY"]
    if not entries:
        print("  No entries to filter.")
        return

    # Compute augmented signals on rolling windows of klines
    aug = AugmentedSignals(lookback_bars=120)

    print(f"\n{'='*70}")
    print(f"AUGMENTED SIGNAL FILTER — {label}")
    print(f"{'='*70}")
    print(f"  Base entries: {len(entries)}")
    print()
    print(f"  {'Time':>5s}  {'Gold':>8s}  {'S(t)':>7s}  {'Vol':>6s}  {'Mom':>6s}  {'Str':>6s}  Filter")
    print(f"  {'-----':>5s}  {'------':>8s}  {'------':>7s}  {'-----':>6s}  {'-----':>6s}  {'-----':>6s}  ------")

    passed = 0
    filtered = 0

    for entry in entries:
        entry_time = entry["time"]
        entry_gold = entry["gold"]
        entry_s = entry["S"]

        # Find the kline index corresponding to this entry time
        entry_idx = None
        for i, kl in enumerate(klines):
            dt = datetime.fromtimestamp(kl[0] / 1000, tz=UTC).astimezone(tz)
            if dt.strftime("%H:%M") == entry_time:
                entry_idx = i
                break

        if entry_idx is None or entry_idx < 50:
            print(f"  {entry_time:>5s}  ${entry_gold:>7.1f}  {entry_s:>7.3f}  {'?':>6s}  {'?':>6s}  {'?':>6s}  SKIP (not enough data)")
            continue

        # Use klines up to this point for augmented signal computation
        window_klines = klines[max(0, entry_idx - 120):entry_idx + 1]
        aug_state = aug.compute_from_klines(window_klines)

        allowed, reason = aug_state.entry_allowed()
        vol_a = aug_state.volatility_action
        mom_a = aug_state.momentum_action
        str_a = aug_state.structure_action

        if allowed:
            status = "PASS"
            passed += 1
        else:
            status = f"BLOCKED ({reason.split(': ')[1]})"
            filtered += 1

        print(f"  {entry_time:>5s}  ${entry_gold:>7.1f}  {entry_s:>7.3f}  {vol_a:>6s}  {mom_a:>6s}  {str_a:>6s}  {status}")

    # Now check P&L of filtered vs unfiltered entries
    print(f"\n  Result: {passed} passed, {filtered} filtered (out of {len(entries)} entries)")

    # Show P&L comparison
    exits = [a for a in alerts if a["type"] == "EXIT"]
    if exits:
        all_pnl = [a["pnl_pct"] for a in exits]
        total_pnl = sum(all_pnl)
        wins = sum(1 for p in all_pnl if p > 0)
        print(f"  Unfiltered P&L: gold {total_pnl:+.3f}% | cert {total_pnl*20:+.1f}% | {wins}/{len(exits)} wins")


def main_with_augmented():
    """Extended main: parameter sweep + augmented signal filter test."""
    main()

    # Re-run with augmented filtering on the production config
    print("\n")
    klines = fetch_klines_today()
    if not klines or len(klines) < 50:
        print("Not enough klines for augmented signal test")
        return

    usdsek = fetch_usdsek_today()
    us10y = 0.04

    # Production config
    prod_config = (0.7, 0.1, 3, 1.0, 0.0, 0.0, 120, 20)
    run_augmented_filter(klines, usdsek, us10y, prod_config, "Production (0.7/0.1/3, win=120)")

    # Also test the noisier config to see how much filtering helps
    noisy_config = (0.5, 0.0, 2, 1.0, 0.0, 0.0, 120, 20)
    run_augmented_filter(klines, usdsek, us10y, noisy_config, "Noisy (0.5/0.0/2, win=120)")


if __name__ == "__main__":
    main_with_augmented()
