from __future__ import annotations

import json, math, os, time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import *

Path(DATA_DIR).mkdir(exist_ok=True)

SP500_FILE = Path(DATA_DIR) / "sp500.csv"
STATE_FILE = Path(DATA_DIR) / "state.json"
OPT_CACHE_DIR = Path(DATA_DIR) / "option_cache"
OPT_CACHE_DIR.mkdir(exist_ok=True)
OPT_VOL_HISTORY = Path(DATA_DIR) / "option_volume_history.csv"


def safe_symbol(sym: str) -> str:
    return sym.replace(".", "-")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def update_sp500_once_daily() -> List[str]:
    today = date.today().isoformat()

    if SP500_FILE.exists():
        old = pd.read_csv(SP500_FILE)
        if not old.empty and str(old.get("updated_at", [""])[0]) == today:
            return old["ticker"].tolist()

    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = sorted(df["Symbol"].map(safe_symbol).unique())
        pd.DataFrame({"ticker": tickers, "updated_at": today}).to_csv(SP500_FILE, index=False)
        return tickers
    except Exception:
        if SP500_FILE.exists():
            return pd.read_csv(SP500_FILE)["ticker"].tolist()
        return sorted(set(WHITELIST))


def get_run_universe() -> List[str]:
    if WHITELIST:
        tickers = WHITELIST
    else:
        tickers = update_sp500_once_daily()

    tickers = [t for t in tickers if t not in BLACKLIST]
    tickers = sorted(set(tickers))

    return tickers[:MAX_TICKERS_PER_RUN]


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def hv30(daily: pd.DataFrame) -> float:
    r = np.log(daily["Close"] / daily["Close"].shift(1)).dropna()
    if len(r) < 30:
        return np.nan
    return float(r.tail(30).std() * math.sqrt(252))


def volume_profile_shelves(df: pd.DataFrame, bins: int = 24, top_n: int = 3) -> List[float]:
    if df.empty:
        return []

    prices = ((df["High"] + df["Low"] + df["Close"]) / 3).dropna()
    vols = df.loc[prices.index, "Volume"].fillna(0)

    if len(prices) < 5 or vols.sum() <= 0:
        return []

    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    idx = np.argsort(hist)[-top_n:][::-1]

    return [float((edges[i] + edges[i + 1]) / 2) for i in idx]


def boll_mid(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["Close"].rolling(n).mean().iloc[-1])


def black_scholes_delta(S, K, T, iv, call=True, r=0.045):
    if S <= 0 or K <= 0 or T <= 0 or not np.isfinite(iv) or iv <= 0:
        return np.nan

    from math import log, sqrt, erf

    d1 = (log(S / K) + (r + iv * iv / 2) * T) / (iv * sqrt(T))
    nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))

    return nd1 if call else nd1 - 1


def pick_expiry(options: List[str], lo: int, hi: int) -> Optional[str]:
    today = date.today()
    candidates = []

    for e in options:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if lo <= dte <= hi:
            candidates.append((abs(dte - ((lo + hi) // 2)), e))

    return sorted(candidates)[0][1] if candidates else None


def cache_path(ticker: str, expiry: str) -> Path:
    return OPT_CACHE_DIR / f"{ticker}_{expiry}.json"


def get_option_chain(ticker: str, expiry: str):
    cp = cache_path(ticker, expiry)

    if cp.exists() and (time.time() - cp.stat().st_mtime) < OPTION_CACHE_MINUTES * 60:
        data = json.loads(cp.read_text())
        return pd.DataFrame(data["calls"]), pd.DataFrame(data["puts"])

    tk = yf.Ticker(ticker)
    ch = tk.option_chain(expiry)

    calls = ch.calls.copy()
    puts = ch.puts.copy()

    cp.write_text(
        json.dumps(
            {"calls": calls.to_dict("records"), "puts": puts.to_dict("records")},
            default=str,
        )
    )

    return calls, puts


def save_option_volume_snapshot(ticker: str, expiry: str, calls: pd.DataFrame, puts: pd.DataFrame):
    rows = []
    ts = date.today().isoformat()

    for side, df in [("CALL", calls), ("PUT", puts)]:
        for _, r in df.iterrows():
            rows.append(
                {
                    "date": ts,
                    "ticker": ticker,
                    "expiry": expiry,
                    "side": side,
                    "strike": float(r["strike"]),
                    "volume": float(r.get("volume", 0) or 0),
                }
            )

    snap = pd.DataFrame(rows)

    if OPT_VOL_HISTORY.exists():
        old = pd.read_csv(OPT_VOL_HISTORY)
        snap = pd.concat([old, snap], ignore_index=True)
        snap = snap.drop_duplicates(
            ["date", "ticker", "expiry", "side", "strike"],
            keep="last",
        )

    snap.to_csv(OPT_VOL_HISTORY, index=False)


def avg_option_volume_20d(ticker: str, expiry: str, side: str, strike: float) -> float:
    if not OPT_VOL_HISTORY.exists():
        return np.nan

    h = pd.read_csv(OPT_VOL_HISTORY)
    q = h[
        (h.ticker == ticker)
        & (h.expiry == expiry)
        & (h.side == side)
        & (h.strike == strike)
    ]

    return float(q.tail(20)["volume"].mean()) if len(q) else np.nan


def iv_rank_from_chains(chains: List[pd.DataFrame]) -> float:
    iv = pd.concat(chains)["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(iv) < 10:
        return np.nan

    cur = float(iv.median())
    lo = float(iv.quantile(0.05))
    hi = float(iv.quantile(0.95))

    if hi <= lo:
        return np.nan

    return 100 * (cur - lo) / (hi - lo)


def atm_pm2(df: pd.DataFrame, S: float) -> pd.DataFrame:
    x = df.copy()
    x["dist"] = (x["strike"] - S).abs()
    return x.sort_values("dist").head(5).copy()


def earnings_within_7d(tk: yf.Ticker) -> bool:
    try:
        cal = tk.calendar

        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            vals = pd.to_datetime(cal.loc["Earnings Date"].dropna()).dt.date.tolist()
        elif isinstance(cal, dict):
            vals = pd.to_datetime(cal.get("Earnings Date", [])).date.tolist()
        else:
            return False

        today = date.today()
        return any(0 <= (d - today).days <= 7 for d in vals)

    except Exception:
        return False


def prescreen(ticker: str, tk: yf.Ticker, daily: pd.DataFrame, monthly: pd.DataFrame) -> Tuple[bool, str]:
    if daily.empty or len(daily) < 60 or monthly.empty or len(monthly) < 60:
        return False, "not enough history"

    S = float(daily["Close"].iloc[-1])
    adv = float(daily["Volume"].tail(30).mean())

    if adv <= 1_000_000 or not (15 <= S <= 600):
        return False, "stock liquidity/price failed"

    if S <= float(ema(monthly["Close"], 50).iloc[-1]):
        return False, "below monthly EMA50"

    if earnings_within_7d(tk):
        return False, "earnings within 7 days"

    return True, "ok"


def analyze_ticker(ticker: str) -> List[str]:
    alerts = []

    tk = yf.Ticker(ticker)

    daily = tk.history(period="1y", interval="1d", auto_adjust=False)
    monthly = tk.history(period="10y", interval="1mo", auto_adjust=False)
    fourh = tk.history(period="90d", interval="1h", auto_adjust=False)

    if not fourh.empty:
        fourh = (
            fourh.resample("4h")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

    ok, why = prescreen(ticker, tk, daily, monthly)

    if not ok:
        print(f"{ticker}: skipped - {why}")
        return alerts

    S = float(daily["Close"].iloc[-1])
    hv = hv30(daily)

    options = list(tk.options or [])

    for label, rng in {"30DTE": (28, 32), "90DTE": (88, 92)}.items():
        expiry = pick_expiry(options, *rng)

        if not expiry:
            continue

        calls, puts = get_option_chain(ticker, expiry)

        if calls.empty and puts.empty:
            continue

        total_volume = (
            calls.get("volume", pd.Series(dtype=float)).fillna(0).sum()
            + puts.get("volume", pd.Series(dtype=float)).fillna(0).sum()
        )

        if total_volume <= 0:
            continue

        save_option_volume_snapshot(ticker, expiry, calls, puts)

        ivr = iv_rank_from_chains([calls, puts])

        if not np.isfinite(ivr) or not (20 < ivr < 80):
            continue

        f = fourh.tail(60).copy()
        m = monthly.copy()

        f["ema9"] = ema(f.Close, 9)
        f["ema21"] = ema(f.Close, 21)
        f["ema50"] = ema(f.Close, 50)
        f["ema200"] = ema(f.Close, 200)

        fourh_shelves = volume_profile_shelves(f, top_n=3)
        monthly_shelves = volume_profile_shelves(m, top_n=3)

        d_atr = float(atr(daily).iloc[-1])
        bbmid = boll_mid(daily)

        month_ema50 = float(ema(m.Close, 50).iloc[-1])
        month_ema200 = float(ema(m.Close, 200).iloc[-1])
        last_month_green = bool(m.Close.iloc[-1] >= m.Open.iloc[-1])

        for side, chain in [("CALL", calls), ("PUT", puts)]:
            sel = atm_pm2(chain, S)

            if sel.empty:
                continue

            premium_total = float((sel["lastPrice"].fillna(0) * sel["volume"].fillna(0) * 100).sum())
            vol = float(sel["volume"].fillna(0).sum())
            oi = float(sel["openInterest"].fillna(0).sum())

            avg20 = np.nanmean(
                [
                    avg_option_volume_20d(ticker, expiry, side, float(k))
                    for k in sel["strike"]
                ]
            )

            med_iv = float(
                sel["impliedVolatility"]
                .replace([np.inf, -np.inf], np.nan)
                .median()
            )

            spread_pct = float(
                (
                    (sel["ask"] - sel["bid"])
                    / ((sel["ask"] + sel["bid"]) / 2)
                )
                .replace([np.inf, -np.inf], np.nan)
                .median()
                * 100
            )

            T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days, 1) / 365
            atm_strike = float(sel.sort_values("dist").iloc[0]["strike"])
            delta = black_scholes_delta(S, atm_strike, T, med_iv, call=(side == "CALL"))
            dte = int(T * 365)

            triggers = []

            avg_ok = np.isfinite(avg20) and avg20 > 0 and vol > 3 * avg20

            if avg_ok and vol > oi and premium_total >= 100_000:
                triggers.append("Unusual options flow")

            near = chain[(chain["strike"] >= S * 0.97) & (chain["strike"] <= S * 1.03)]

            if float(near["openInterest"].fillna(0).sum()) > 20_000:
                triggers.append("Gamma squeeze setup")

            if side == "CALL" and np.isfinite(hv) and med_iv < hv - 0.05:
                triggers.append("IV edge: IV below HV30")

            if side == "PUT" and np.isfinite(hv) and med_iv > hv + 0.05:
                triggers.append("IV edge: IV above HV30")

            if len(f) >= 50 and fourh_shelves:
                shelf = min(fourh_shelves, key=lambda x: abs(x - S))

                if side == "CALL" and S > f.ema50.iloc[-1] and S > shelf and f.ema9.iloc[-1] > f.ema21.iloc[-1]:
                    triggers.append("4H bullish confluence")

                if side == "PUT" and S < f.ema50.iloc[-1] and S < shelf and f.ema9.iloc[-1] < f.ema21.iloc[-1]:
                    triggers.append("4H bearish confluence")

            if monthly_shelves:
                if side == "CALL" and any(0 <= (S - sh) / S <= 0.08 for sh in monthly_shelves):
                    triggers.append("Monthly support nearby")

                if side == "PUT" and any(0 <= (sh - S) / S <= 0.08 for sh in monthly_shelves):
                    triggers.append("Monthly resistance nearby")

            score = len(triggers)

            if score >= ALERT_SCORE_MIN:
                direction = "LONG CALL" if side == "CALL" else "LONG PUT / BEARISH PUT WATCH"

                lines = [
                    f"🚨 {direction} ALERT: {ticker} {label}",
                    f"Price ${S:.2f} | Exp {expiry} | DTE {dte} | Score {score}/5",
                    f"ATM±2 strikes: {', '.join(map(lambda x: str(round(x, 2)), sel['strike'].tolist()))}",
                    f"Delta {delta:.2f} | IV {med_iv*100:.1f}% | IV Rank {ivr:.1f}% | HV30 {hv*100:.1f}%",
                    f"Vol/OI {vol:.0f}/{oi:.0f} | Premium ${premium_total:,.0f} | Spread {spread_pct:.1f}%",
                    f"4H EMA9/21/50/200: {f.ema9.iloc[-1]:.2f}/{f.ema21.iloc[-1]:.2f}/{f.ema50.iloc[-1]:.2f}/{f.ema200.iloc[-1]:.2f}",
                    f"Daily BB Mid {bbmid:.2f} | ATR14 {d_atr:.2f}",
                    f"Monthly EMA50/200 {month_ema50:.2f}/{month_ema200:.2f} | Last candle {'green' if last_month_green else 'red'}",
                    f"Monthly VP shelves: {', '.join(f'${x:.2f}' for x in monthly_shelves)}",
                    "Triggers: " + "; ".join(triggers),
                    "Risk note: verify fill, news, earnings, and full chain before trading.",
                ]

                alerts.append("\n".join(lines))

    return alerts


def send_alert(text: str) -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": text[:3900],
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            return
        except Exception as e:
            print(f"Telegram failed: {e}")

    print("\n" + text + "\n")

    if not webhook:
        print("\n" + text + "\n")
        return

    try:
        requests.post(webhook, json={"content": text, "text": text}, timeout=10)
    except Exception as e:
        print(f"Webhook failed: {e}\n{text}")


def main():
    batch = get_run_universe()

    print(f"Running {len(batch)} tickers:")
    print(batch)

    total_alerts = 0

    for ticker in batch:
        try:
            print(f"Checking {ticker}...")
            alerts = analyze_ticker(ticker)

            for alert in alerts:
                total_alerts += 1
                send_alert(alert)

        except Exception as e:
            print(f"{ticker}: {type(e).__name__}: {e}")

        time.sleep(SLEEP_BETWEEN_TICKERS)

    print(f"Done. Total alerts: {total_alerts}")


if __name__ == "__main__":
    main()