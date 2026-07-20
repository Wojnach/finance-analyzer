"""Microstructure-features XGBoost walk-forward backtest — input-side pivot.

Clone of scripts/xgb_backtest.py (same grid, resume keys, row schema, XGB
params, no-leakage discipline) varying only the INPUT features, per
docs/plans/2026-07-20-microstructure-pivot-design.md:

    --feature-set kline  Groups A+B — taker-flow + realized-vol structure,
                         computed from the kline columns lb.fetch_klines_1h
                         already returns (tbb/tbq/n/qv)
    --feature-set full   kline + the xgb_backtest indicator baseline
                         (rsi, macd_hist, ema_gap_pct, bb_pos, vol_ratio,
                         chg24, lr1/3/6/12, fng)
    --feature-set stage1 kline + design Group C perp positioning (funding via
                         REST fapi/v1/fundingRate, OI + long/short ratios via
                         data.binance.vision bulk daily metrics) + Group D
                         cross-asset (DXY/SPY yfinance 1h with staleness
                         flags, BTC exogenous returns, gold/silver ratio z on
                         metals runs). External series cached resumably under
                         data/micro_cache/.
    --prefetch-only      only build the data/micro_cache/ caches for the
                         requested window, then exit — lets the download run
                         separately (e.g. on herc2) before the backtest.

Rows land as model "xgbmicro-<feature_set>-<h>h" — schema-compatible with
the existing matrix, llm_backtest.py --score works unchanged.

Usage:
    python scripts/micro_backtest.py --start 2025-08-01 --end 2026-07-11 \
        --interval 1h --step-hours 8 --tickers BTC-USD,ETH-USD \
        --feature-set kline --out data/xgb_backtest_results.jsonl
    python scripts/micro_backtest.py --prefetch-only --start ... --end ... \
        --tickers BTC-USD,ETH-USD
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_backtest as lb  # noqa: E402 — import is side-effect free (__main__ guard)
from portfolio.signal_utils import ema, rsi  # noqa: E402 — repo root on path via lb

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    eval_metric="logloss",
    random_state=0,
)
MIN_TRAIN_ROWS = 200
MIN_HIST = 120  # mirrors build_context's 120-candle minimum
Z_WIN = 100  # rolling window for z-scored features
FEATURE_SETS = ("kline", "full", "stage1")

CACHE_DIR = lb.REPO / "data" / "micro_cache"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
METRICS_ZIP = (
    "https://data.binance.vision/data/futures/um/daily/metrics/"
    "{sym}/{sym}-metrics-{day}.zip"
)
METRICS_COLS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
# staleness caps for the as-of merges (design §3 NaN policy)
FUNDING_TOL = pd.Timedelta(hours=8, minutes=5)  # settles every 8h (+5m slack)
METRICS_TOL = pd.Timedelta(minutes=15)  # 5-min rows: forward-fill <= 15 min
XASSET_STALE = pd.Timedelta(hours=2)  # DXY/SPY older than this -> *_stale=1


def cast_micro(df: pd.DataFrame) -> pd.DataFrame:
    # fetch_klines_1h keeps qv/n/tbb/tbq as strings (only OHLCV cast)
    for c in ("qv", "n", "tbb", "tbq"):
        df[c] = df[c].astype(float)
    return df


def _z(s: pd.Series, win: int = Z_WIN) -> pd.Series:
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


# ---------------------------------------------------------------------------
# Stage-1 external data — cached, resumable, idempotent (data/micro_cache/).
# Crypto tickers use SPOT klines (lb.TICKERS) but funding/metrics only exist
# for the USDT-M PERP; the symbol string is identical (BTC-USD -> BTCUSDT on
# FAPI), so Group C attaches the perp's positioning to the spot row — a
# cross-market feature by construction (design §3 note + §6 risk 2). Metals
# (XAUUSDT/XAGUSDT) are FAPI-native, no mismatch.
# ---------------------------------------------------------------------------


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_gz(path: Path):
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else None


def _save_gz(path: Path, obj) -> None:
    _atomic_bytes(path, gzip.compress(json.dumps(obj).encode()))


def _ms_int(dt) -> list:
    # pandas 3 keeps non-nanosecond datetime units (yf gives [s], parsed CSV
    # strings give [us]) — normalize to ms explicitly before caching
    return [int(v) for v in pd.DatetimeIndex(dt).as_unit("ms").astype("int64")]


def _http_get(url, params=None, ok404=False, timeout=30):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if ok404 and r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Settled funding rates via REST startTime paging (design §1.3).

    Cache data/micro_cache/funding_<SYM>.json records the requested coverage
    window, so reruns are no-ops and window extensions fetch only the
    missing head/tail (resumable + idempotent).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"funding_{symbol}.json"
    cache = json.loads(path.read_text()) if path.exists() else None
    if cache is None or start_ms < cache["start_ms"] or end_ms > cache["end_ms"]:
        rows = {int(t): r for t, r in (cache["rows"] if cache else [])}
        if cache is None:
            spans = [(start_ms, end_ms)]
        else:
            spans = []
            if start_ms < cache["start_ms"]:
                spans.append((start_ms, cache["start_ms"] - 1))
            if end_ms > cache["end_ms"]:
                spans.append((cache["end_ms"] + 1, end_ms))
        fetched = 0
        for s, e in spans:
            cur = s
            while cur <= e:
                batch = _http_get(
                    FUNDING_URL,
                    params={
                        "symbol": symbol,
                        "startTime": cur,
                        "endTime": e,
                        "limit": 1000,
                    },
                ).json()
                if not isinstance(batch, list):
                    raise SystemExit(f"funding {symbol}: {batch}")
                if not batch:
                    break
                for d in batch:
                    rows[int(d["fundingTime"])] = float(d["fundingRate"])
                fetched += len(batch)
                if len(batch) < 1000:
                    break
                cur = int(batch[-1]["fundingTime"]) + 1
                time.sleep(0.15)
        cache = {
            "symbol": symbol,
            "start_ms": min(start_ms, cache["start_ms"]) if cache else start_ms,
            "end_ms": max(end_ms, cache["end_ms"]) if cache else end_ms,
            "rows": sorted([t, r] for t, r in rows.items()),
        }
        _atomic_bytes(path, json.dumps(cache).encode())
        print(f"funding {symbol}: +{fetched} settlements fetched", flush=True)
    else:
        print(
            f"funding {symbol}: cache hit ({len(cache['rows'])} settlements)",
            flush=True,
        )
    pairs = [(t, r) for t, r in cache["rows"] if start_ms <= t <= end_ms]
    if not pairs:
        raise SystemExit(f"funding {symbol}: no settlements in window (listed yet?)")
    df = pd.DataFrame(
        {"rate": [r for _, r in pairs]},
        index=pd.to_datetime([t for t, _ in pairs], unit="ms", utc=True),
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_metrics(
    symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    """5-min OI/long-short metrics from data.binance.vision daily zips
    (design §1.4/§1.5 — the only backfill path, REST caps at ~30 days).

    Per-day download state lives in metrics_<SYM>_meta.json, concatenated
    rows in metrics_<SYM>.json.gz (no pyarrow on the Deck venv, so json.gz
    instead of parquet). Days already fetched (or known-missing) are never
    re-downloaded; recent days (< 2 days old, dumps not yet published) are
    skipped without being marked so a later run retries them.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / f"metrics_{symbol}_meta.json"
    data_path = CACHE_DIR / f"metrics_{symbol}.json.gz"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"days": {}}
    data = _load_gz(data_path) or {"create_time": [], **{c: [] for c in METRICS_COLS}}

    def _save():
        order = np.argsort(np.asarray(data["create_time"]))
        for k in list(data):
            data[k] = [data[k][i] for i in order]
        _save_gz(data_path, data)
        _atomic_bytes(meta_path, json.dumps(meta).encode())

    today = pd.Timestamp.now(tz="UTC").normalize()
    downloaded = missing = hit = 0
    dirty = False
    for day in pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D"):
        key = day.strftime("%Y-%m-%d")
        if meta["days"].get(key) in ("ok", "missing"):
            hit += 1
            continue
        if day >= today - pd.Timedelta(days=1):
            continue  # dump not published yet — retry on a later run
        r = _http_get(METRICS_ZIP.format(sym=symbol, day=key), ok404=True)
        if r is None:
            missing += 1
            if day < today - pd.Timedelta(days=3):
                meta["days"][key] = "missing"  # permanent gap in the bulk data
                dirty = True
            print(f"metrics {symbol} {key}: missing (skipped)", flush=True)
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv = pd.read_csv(z.open(z.namelist()[0]))
        ct = csv["create_time"]
        ts = (
            pd.to_datetime(ct, unit="ms", utc=True)
            if pd.api.types.is_numeric_dtype(ct)
            else pd.to_datetime(ct, utc=True)
        )
        data["create_time"].extend(_ms_int(ts))
        for c in METRICS_COLS:
            data[c].extend(csv[c].astype(float).tolist())
        meta["days"][key] = "ok"
        dirty = True
        downloaded += 1
        if downloaded % 30 == 0:
            _save()  # checkpoint so an interrupted download resumes
        time.sleep(0.05)
    if dirty:
        _save()
    print(
        f"metrics {symbol}: {downloaded} days downloaded, {hit} cached, "
        f"{missing} missing ({len(data['create_time'])} rows total)",
        flush=True,
    )
    if not data["create_time"]:
        raise SystemExit(f"metrics {symbol}: no rows in cache for window")
    idx = pd.to_datetime(data["create_time"], unit="ms", utc=True)
    df = pd.DataFrame(
        {c: np.asarray(data[c], dtype=float) for c in METRICS_COLS}, index=idx
    ).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def fetch_yf_closes(key: str, symbol: str, start_ts, end_ts) -> pd.Series:
    """yfinance 1h closes, cached to data/micro_cache/yf_<key>.json.gz.

    Cache hit when the recorded coverage spans the request; otherwise the
    window is refetched (one request) and merged, keeping the union coverage
    so herc2-prefetched caches serve later Deck runs offline.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"yf_{key}.json.gz"
    now = pd.Timestamp.now(tz="UTC")
    # yfinance serves ~730 days of hourly bars (design §1.8) — clamp
    want_s = max(pd.Timestamp(start_ts), now - pd.Timedelta(days=728))
    want_e = min(pd.Timestamp(end_ts), now)
    cache = _load_gz(path)
    s_ms, e_ms = int(want_s.timestamp() * 1000), int(want_e.timestamp() * 1000)
    if cache and cache["start_ms"] <= s_ms and cache["end_ms"] >= e_ms:
        print(f"yf {key}: cache hit ({len(cache['ts'])} bars)", flush=True)
    else:
        df = lb.fetch_klines_yf(
            symbol,
            want_s.strftime("%Y-%m-%d"),
            (want_e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1h",
        )
        rows = dict(zip(cache["ts"], cache["close"])) if cache else {}
        for t, c in zip(_ms_int(df.index), df["close"].tolist()):
            rows[t] = float(c)
        cache = {
            "symbol": symbol,
            "start_ms": min(s_ms, cache["start_ms"]) if cache else s_ms,
            "end_ms": max(e_ms, cache["end_ms"]) if cache else e_ms,
            "ts": sorted(rows),
            "close": [rows[t] for t in sorted(rows)],
        }
        _save_gz(path, cache)
        print(f"yf {key}: fetched, cache now {len(cache['ts'])} bars", flush=True)
    idx = pd.to_datetime(cache["ts"], unit="ms", utc=True)
    s = pd.Series(cache["close"], index=idx, dtype=float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.loc[
        (s.index >= pd.Timestamp(start_ts)) & (s.index <= pd.Timestamp(end_ts))
    ]


# ---------------------------------------------------------------------------
# Stage-1 feature builders. Each returns a table indexed by the timestamp the
# record became PUBLIC (publication-lag shifts per design §4.6); merging back
# onto candles is a plain as-of-backward lookup with a staleness cap.
# ---------------------------------------------------------------------------


def _ret_24h(lr: pd.Series, cap_hours: int = 96) -> pd.Series:
    # NO-LEAKAGE: last record with timestamp <= t-24h (searchsorted right-1),
    # same asof pattern as feature_frame's chg24. Gaps wider than cap -> NaN.
    idx = lr.index
    pos = idx.searchsorted(idx - pd.Timedelta(hours=24), side="right") - 1
    okpos = np.clip(pos, 0, None)
    ok = (pos >= 0) & ((idx - idx[okpos]) <= pd.Timedelta(hours=cap_hours))
    v = lr.to_numpy()
    out = np.full(len(v), np.nan)
    out[ok] = v[ok] - v[okpos[ok]]
    return pd.Series(out, index=idx)


def funding_features(fdf: pd.DataFrame) -> pd.DataFrame:
    # NO-LEAKAGE: each settlement's delta/z use only settlements at or before
    # its own fundingTime (diff/rolling end at the current row); the rate is
    # public at fundingTime, so no extra lag shift (design §4.6).
    r = fdf["rate"]
    out = pd.DataFrame(index=fdf.index)
    out["funding"] = r
    out["funding_delta"] = r.diff()
    m30 = r.rolling("30D", min_periods=30).mean()
    s30 = r.rolling("30D", min_periods=30).std()
    out["funding_z30"] = (r - m30) / s30
    return out


def metrics_features(mdf: pd.DataFrame) -> pd.DataFrame:
    # Group C from the 5-min metrics rows. All rolling/diff ops end at the
    # row's own create_time (causal); step-count diffs are gap-guarded so a
    # missing bulk day can't silently become a bogus 1h/24h delta.
    oi = mdf["sum_open_interest"]
    out = pd.DataFrame(index=mdf.index)
    out["oi_norm"] = oi / oi.rolling("7D", min_periods=288).mean()
    loi = np.log(oi.where(oi > 0))
    tgap = mdf.index.to_series()
    out["oi_d1h"] = (loi - loi.shift(12)).where(
        tgap.diff(12) <= pd.Timedelta(minutes=70)
    )
    out["oi_d24h"] = (loi - loi.shift(288)).where(
        tgap.diff(288) <= pd.Timedelta(hours=24, minutes=30)
    )
    out["taker_ls_ratio"] = mdf["sum_taker_long_short_vol_ratio"].rolling("1h").mean()
    out["top_trader_ls"] = mdf["sum_toptrader_long_short_ratio"]
    out["global_ls"] = mdf["count_long_short_ratio"]
    # NO-LEAKAGE: +5 min publication lag (design §4.6) — a 12:00 sample is
    # only visible to candles at/after 12:05.
    out.index = out.index + pd.Timedelta(minutes=5)
    return out


def xasset_features(close: pd.Series, prefix: str) -> pd.Series:
    # DXY/SPY from yfinance hourly bars (bar-START indexed).
    # NO-LEAKAGE: a bar's close only exists once the bar ENDS — index is
    # shifted +1h so the as-of merge can only pick fully closed bars
    # (design §4.6 "last closed hourly bar").
    lr = np.log(close)
    out = pd.DataFrame(index=close.index)
    out[f"{prefix}_r1"] = lr.diff()
    out[f"{prefix}_r24"] = _ret_24h(lr)
    out.index = out.index + pd.Timedelta(hours=1)
    out["asof_ts"] = out.index  # for the *_stale flag at merge time
    return out


def exo_kline_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # BTC exogenous returns from Binance klines at the run interval.
    # NO-LEAKAGE caveat: indexed by open_time reading the as-of candle's
    # close — byte-identical to the convention the row's own features and
    # outcomes use (design §6 risk 3), so the information basis matches.
    lr = np.log(df["close"])
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_r1"] = lr.diff()
    out[f"{prefix}_r24"] = _ret_24h(lr)
    return out


def gs_features(xau_close: pd.Series, xag_close: pd.Series) -> pd.DataFrame:
    # gold/silver ratio z from the XAU/XAG kline closes (design Group D);
    # same open_time as-of convention as exo_kline_features.
    common = xau_close.index.intersection(xag_close.index)
    ratio = xau_close.loc[common] / xag_close.loc[common]
    return pd.DataFrame({"gold_silver_ratio_z": _z(ratio)}, index=common)


def build_stage1_extras(wanted, frames, interval, iv_h, pad, end):
    # external window: pad extra days so 7d OI mean / 30d funding z / 24h
    # returns are warm before the first training candle
    ext_s = pad - pd.Timedelta(days=8)
    ext_e = end + pd.Timedelta(days=8)
    s_ms, e_ms = int(ext_s.timestamp() * 1000), int(ext_e.timestamp() * 1000)
    kline_end_ms = int((end + pd.Timedelta(days=8)).timestamp() * 1000)
    dxy = xasset_features(fetch_yf_closes("DXY", "DX-Y.NYB", ext_s, ext_e), "dxy")
    spy = xasset_features(fetch_yf_closes("SPY", "SPY", ext_s, ext_e), "spy")
    # BTC exogenous on every row (incl. BTC's own — a pooled cross-ticker
    # model needs identical columns on all rows)
    if "BTC-USD" in frames:
        bdf = frames["BTC-USD"]
    else:
        bdf = lb.fetch_klines_1h(
            lb.TICKERS["BTC-USD"],
            int(pad.timestamp() * 1000),
            kline_end_ms,
            base_url=lb.BINANCE_KLINES,
            interval=interval,
        )
    btc = exo_kline_features(bdf, "btc")
    # gold/silver ratio z only when the run holds metals rows; the column is
    # then attached to EVERY ticker in the run (rectangular pooled matrix),
    # crypto-only runs omit it entirely
    gs = None
    if any(t in lb.FAPI_TICKERS for t in wanted):
        closes = {}
        for t in ("XAU-USD", "XAG-USD"):
            closes[t] = (
                frames[t]["close"]
                if t in frames
                else lb.fetch_klines_1h(
                    lb.FAPI_TICKERS[t],
                    int(pad.timestamp() * 1000),
                    kline_end_ms,
                    base_url=lb.BINANCE_FAPI_KLINES,
                    interval=interval,
                )["close"]
            )
        gs = gs_features(closes["XAU-USD"], closes["XAG-USD"])
    extras = {}
    for tick in wanted:
        # spot ticker -> same symbol string on FAPI perp (BTCUSDT/ETHUSDT);
        # metals already FAPI-native
        sym = lb.TICKERS.get(tick) or lb.FAPI_TICKERS.get(tick)
        if sym is None:
            raise SystemExit(f"{tick}: no FAPI perp mapping for stage1")
        fu = funding_features(fetch_funding(sym, s_ms, e_ms))
        me = metrics_features(fetch_metrics(sym, ext_s, ext_e))
        extras[tick] = {
            "funding": fu,
            "metrics": me,
            "dxy": dxy,
            "spy": spy,
            "btc": btc,
            "gs": gs,
            "iv_h": iv_h,
        }
    return extras


def prefetch(wanted, pad, end):
    ext_s = pad - pd.Timedelta(days=8)
    ext_e = end + pd.Timedelta(days=8)
    s_ms, e_ms = int(ext_s.timestamp() * 1000), int(ext_e.timestamp() * 1000)
    for tick in wanted:
        sym = lb.TICKERS.get(tick) or lb.FAPI_TICKERS.get(tick)
        if sym is None:
            raise SystemExit(f"{tick}: no FAPI perp mapping for stage1 prefetch")
        fetch_funding(sym, s_ms, e_ms)
        fetch_metrics(sym, ext_s, ext_e)
    fetch_yf_closes("DXY", "DX-Y.NYB", ext_s, ext_e)
    fetch_yf_closes("SPY", "SPY", ext_s, ext_e)
    # klines are intentionally NOT cached — REST refetch at backtest time is
    # cheap (~5 requests per symbol-year)
    print("PREFETCH COMPLETE — cache contents:", flush=True)
    for p in sorted(CACHE_DIR.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB", flush=True)


def feature_frame(
    df: pd.DataFrame, fng: dict, feature_set: str, extras: dict | None = None
) -> pd.DataFrame:
    # NO-LEAKAGE: every column is causal — ewm/rolling/shift at candle t use
    # only candles with open_time <= t. Same information basis as
    # build_context, which likewise reads the as-of candle's close at t.
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    # Group A — kline microstructure (design §3)
    taker_imb = (2 * df["tbb"] / vol - 1).where(vol > 0)
    r4 = taker_imb.rolling(4).mean()
    r24 = taker_imb.rolling(24).mean()
    f["taker_imb"] = taker_imb
    f["taker_imb_r4"] = r4
    f["taker_imb_r24"] = r24
    f["taker_imb_z"] = _z(r4)
    f["flow_accel"] = r4 - r24
    f["trade_intensity_z"] = _z(df["n"])
    f["avg_trade_size_z"] = _z((vol / df["n"]).where(df["n"] > 0))
    f["vol_accel"] = np.log(vol.rolling(4).mean() / vol.rolling(24).mean())
    rng = high - low
    f["clv"] = ((2 * close - high - low) / rng).where(rng > 0, 0.0)
    # Group B — realized-vol structure (design §3)
    lr1 = np.log(close / close.shift(1))
    f["rv_ratio"] = lr1.rolling(6).std() / lr1.rolling(48).std()
    pk = np.sqrt((np.log(high / low) ** 2).rolling(24).mean() / (4 * np.log(2)))
    f["pk_vol_z"] = _z(pk)
    f["ret_skew24"] = lr1.rolling(24).skew()
    if feature_set == "full":
        # indicator baseline — copied verbatim from xgb_backtest.feature_frame
        f["rsi"] = rsi(close, 14)
        macd_line = ema(close, 12) - ema(close, 26)
        f["macd_hist"] = macd_line - ema(macd_line, 9)
        f["ema_gap_pct"] = (ema(close, 9) / ema(close, 21) - 1) * 100
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        f["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)
        f["vol_ratio"] = vol / vol.rolling(20).mean()
        # NO-LEAKAGE: last close with open_time <= t-24h, same asof lookup as
        # build_context's day_ago line
        p24 = df.index.searchsorted(df.index - pd.Timedelta(hours=24), side="right") - 1
        chg = (close.to_numpy() / close.to_numpy()[np.clip(p24, 0, None)] - 1) * 100
        chg[p24 < 0] = 0.0
        f["chg24"] = chg
        for n in (1, 3, 6, 12):
            f[f"lr{n}"] = np.log(close / close.shift(n))
        # F&G for date D is published at the start of day D — known intraday at t
        f["fng"] = [fng.get(d, 50) for d in df.index.date]
    if feature_set == "stage1":
        # Groups C+D (design §3), merged as-of BACKWARD per ticker.
        # NO-LEAKAGE: reindex(method="ffill", tolerance=cap) attaches to the
        # candle at open_time t only records stamped <= t — the extras tables
        # are already publication-lag shifted (metrics +5 min, yf bars +1h to
        # bar close, funding at fundingTime), and anything older than the
        # staleness cap becomes NaN so the valid mask drops the row.
        idx = f.index
        cand_tol = pd.Timedelta(hours=2 * extras["iv_h"])
        tables = [
            (extras["funding"], FUNDING_TOL),
            (extras["metrics"], METRICS_TOL),
            (extras["btc"], cand_tol),
        ]
        if extras["gs"] is not None:
            tables.append((extras["gs"], cand_tol))
        for tbl, tol in tables:
            m = tbl.reindex(idx, method="ffill", tolerance=tol)
            for col in m.columns:
                f[col] = m[col]
        for key in ("dxy", "spy"):
            # design §3 NaN policy exception: DXY/SPY forward-fill across
            # session gaps with NO cap + a *_stale indicator, so 24/7 rows
            # aren't discarded off-session/weekends
            m = extras[key].reindex(idx, method="ffill")
            f[f"{key}_r1"] = m[f"{key}_r1"]
            f[f"{key}_r24"] = m[f"{key}_r24"]
            f[f"{key}_stale"] = (
                (pd.Series(idx, index=idx) - m["asof_ts"]) > XASSET_STALE
            ).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def label_arrays(df: pd.DataFrame, horizons, iv_h: float, sessions: bool) -> dict:
    # Label = future close > basis close, same outcome semantics as
    # llm_backtest. resolve = CLOSE time of the future candle
    # (open_time + interval): the label only exists once that candle closed.
    close = df["close"].to_numpy()
    idx = df.index
    out = {}
    for h in horizons:
        if sessions:
            n = max(1, round(h / iv_h * 7 / 24)) if h >= 24 else max(1, round(h / iv_h))
            fpos = np.arange(len(df)) + n
        else:
            fpos = idx.searchsorted(idx + pd.Timedelta(hours=h), side="left")
        ok = np.flatnonzero(fpos < len(df))
        y = np.zeros(len(df), dtype=bool)
        y[ok] = close[fpos[ok]] > close[ok]
        resolve = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
        resolve.iloc[ok] = idx[fpos[ok]] + pd.Timedelta(hours=iv_h)
        out[h] = (y, resolve)
    return out


def run(args):
    from xgboost import XGBClassifier

    t_run = time.time()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    fng = lb.fetch_fng() if args.feature_set == "full" else {}
    horizons = lb.INTERVAL_HORIZONS[args.interval]
    step_h = args.step_hours or lb.INTERVAL_STEP_HOURS[args.interval]
    iv_h = lb._interval_hours(args.interval)
    wanted = [t.strip() for t in args.tickers.split(",")]
    # pad beyond llm_backtest's 200-candle warmup so the expanding training
    # window has resolved labels before the first grid stamp
    pad = start - max(
        pd.Timedelta(days=args.train_days), pd.Timedelta(hours=200 * iv_h)
    )
    if args.prefetch_only:
        prefetch(wanted, pad, end)
        return
    frames = {}
    for tick in wanted:
        if tick in lb.TICKERS:
            sym, url = lb.TICKERS[tick], lb.BINANCE_KLINES
        elif tick in lb.FAPI_TICKERS:
            sym, url = lb.FAPI_TICKERS[tick], lb.BINANCE_FAPI_KLINES
        elif tick in lb.YF_TICKERS:
            raise SystemExit(
                f"{tick}: yfinance candles carry no taker-buy/trade-count fields — "
                "micro feature sets need Binance tickers"
            )
        else:
            raise SystemExit(f"unknown ticker {tick}")
        frames[tick] = cast_micro(
            lb.fetch_klines_1h(
                sym,
                int(pad.timestamp() * 1000),
                int((end + pd.Timedelta(days=8)).timestamp() * 1000),
                base_url=url,
                interval=args.interval,
            )
        )
        print(f"{tick}: {len(frames[tick])} {args.interval} candles", flush=True)

    extras = (
        build_stage1_extras(wanted, frames, args.interval, iv_h, pad, end)
        if args.feature_set == "stage1"
        else {}
    )

    feats, fmats, valid, labels = {}, {}, {}, {}
    for tick in frames:
        feats[tick] = feature_frame(
            frames[tick], fng, args.feature_set, extras.get(tick)
        )
        fmats[tick] = feats[tick].to_numpy(dtype=float)
        bv = np.isfinite(fmats[tick]).all(axis=1)
        bv[: MIN_HIST - 1] = False
        valid[tick] = bv
        labels[tick] = label_arrays(frames[tick], horizons, iv_h, tick in lb.YF_TICKERS)

    # grid construction mirrors llm_backtest.run
    cases = []
    at = start
    while at <= end:
        for tick in frames:
            pos = frames[tick].index.searchsorted(at, side="right") - 1
            if tick in lb.YF_TICKERS:
                # only timestamps that land on an actual session candle
                if pos < 0 or (at - frames[tick].index[pos]) > pd.Timedelta(hours=iv_h):
                    continue
            if pos + 1 < MIN_HIST:
                continue
            if tick in lb.YF_TICKERS:
                outs = {
                    f"outcome_{h}h_pct": lb.outcome_sessions(
                        frames[tick],
                        at,
                        (
                            max(1, round(h / iv_h * 7 / 24))
                            if h >= 24
                            else max(1, round(h / iv_h))
                        ),
                    )
                    for h in horizons
                }
            else:
                outs = {
                    f"outcome_{h}h_pct": lb.outcome_at(frames[tick], at, h)
                    for h in horizons
                }
            if outs[f"outcome_{horizons[-1]}h_pct"] is None:
                continue
            outs = {
                k: (round(v, 3) if v is not None else None) for k, v in outs.items()
            }
            cases.append({"at": at.isoformat(), "ticker": tick, "pos": pos, **outs})
        at += pd.Timedelta(hours=step_h)
    print(f"{len(cases)} cases x {len(horizons)} horizons", flush=True)

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["model"], r.get("interval", "1h"), r["at"], r["ticker"]))
        print(f"resume: {len(done)} results already present", flush=True)

    xgb_params = dict(XGB_PARAMS, n_jobs=args.n_jobs)

    def train(at_ts, h):
        Xs, ys = [], []
        for tick in frames:
            y, resolve = labels[tick][h]
            # NO-LEAKAGE / WALK-FORWARD: expanding window — a row at candle t
            # trains only when its label's future candle CLOSED at or before
            # `at` (resolve <= at; NaT compares False). Features at t already
            # use only candles with open_time <= t.
            m = valid[tick] & (resolve <= at_ts).to_numpy()
            Xs.append(fmats[tick][m])
            ys.append(y[m])
        X, yv = np.concatenate(Xs), np.concatenate(ys)
        if len(yv) < MIN_TRAIN_ROWS or len(np.unique(yv)) < 2:
            return None
        clf = XGBClassifier(**xgb_params)
        clf.fit(X, yv)
        return clf

    written = 0
    with out_path.open("a") as fh:
        model_cache = {}
        cache_at = None
        for c in cases:
            if c["at"] != cache_at:
                model_cache, cache_at = {}, c["at"]
            at_ts = pd.Timestamp(c["at"])
            for h in horizons:
                name = f"xgbmicro-{args.feature_set}-{h}h"
                if (name, args.interval, c["at"], c["ticker"]) in done:
                    continue
                t0 = time.time()
                if h not in model_cache:
                    model_cache[h] = train(at_ts, h)
                clf = model_cache[h]
                x = fmats[c["ticker"]][c["pos"]]
                if clf is None or not np.isfinite(x).all():
                    vote, conf = "ABSTAIN", None
                else:
                    p_up = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
                    vote = "BUY" if p_up > 0.5 else "SELL"
                    conf = round(max(p_up, 1 - p_up), 3)
                row = {
                    "model": name,
                    "interval": args.interval,
                    "arm": "A",
                    "at": c["at"],
                    "ticker": c["ticker"],
                    "vote": vote,
                    "conf": conf,
                    "outcome_pct": c[f"outcome_{horizons[-1]}h_pct"],
                    **{k: c[k] for k in c if k.startswith("outcome_")},
                    "secs": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                written += 1
                if written % 100 == 0:
                    print(f"{written} rows, at={c['at']}", flush=True)
    print(
        f"MICRO BACKTEST COMPLETE — {written} rows in {time.time() - t_run:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-02-01")
    p.add_argument("--end", default="2026-07-11")
    p.add_argument("--step-hours", type=int, default=0)
    p.add_argument("--interval", default="1h", choices=list(lb.INTERVAL_HORIZONS))
    p.add_argument("--tickers", default="BTC-USD,ETH-USD")
    p.add_argument("--feature-set", default="kline", choices=list(FEATURE_SETS))
    p.add_argument("--out", default="data/xgb_backtest_results.jsonl")
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--n-jobs", type=int, default=XGB_PARAMS["n_jobs"])
    p.add_argument("--prefetch-only", action="store_true")
    run(p.parse_args())
