from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    ALERT_THRESHOLD,
    BLACKLIST,
    DAILY_PERIOD,
    DATA_DIR,
    ENTRY_BUFFER,
    ENTRY_LOOKBACK_BARS,
    HOURLY_PERIOD,
    INTRADAY_PERIOD,
    MAX_PRICE,
    MAX_RISK_PCT,
    MAX_TICKERS_PER_RUN,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    POC_BINS,
    POC_LOOKBACK_BARS,
    PT1_R_MULTIPLE,
    PT2_R_MULTIPLE,
    REQUIRE_DAILY_TREND,
    REQUIRE_HOURLY_TREND,
    RS_LOOKBACK_DAYS,
    SCORE_WEIGHTS,
    SECTOR_THEME_MAP,
    SLEEP_BETWEEN_TICKERS,
    STATE_RETENTION_DAYS,
    STOP_BUFFER,
    STOP_LOOKBACK_BARS,
    WHITELIST,
)


# ============================================================
# FILES AND STATE
# ============================================================

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

STATE_FILE = Path(DATA_DIR) / "five_point_signal_state.json"


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    """Return an exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()


def bullish_trend(df: pd.DataFrame) -> bool:
    """
    Bullish trend requirements:

    1. Price is above EMA 9.
    2. EMA 9 is above EMA 21.
    3. EMA 9 is rising.
    4. EMA 21 is flat or rising.
    """
    if df.empty or len(df) < 25:
        return False

    close = df["Close"]
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)

    values = [
        close.iloc[-1],
        ema9.iloc[-1],
        ema21.iloc[-1],
        ema9.iloc[-3],
        ema21.iloc[-3],
    ]

    if not all(np.isfinite(value) for value in values):
        return False

    return bool(
        close.iloc[-1] > ema9.iloc[-1] > ema21.iloc[-1]
        and ema9.iloc[-1] > ema9.iloc[-3]
        and ema21.iloc[-1] >= ema21.iloc[-3]
    )


def price_above_ema9_21(df: pd.DataFrame) -> bool:
    """Check whether price is above both EMA 9 and EMA 21."""
    if df.empty or len(df) < 25:
        return False

    close = df["Close"]
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)

    return bool(
        np.isfinite(close.iloc[-1])
        and np.isfinite(ema9.iloc[-1])
        and np.isfinite(ema21.iloc[-1])
        and close.iloc[-1] > ema9.iloc[-1]
        and close.iloc[-1] > ema21.iloc[-1]
    )


# ============================================================
# DATA FUNCTIONS
# ============================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a yfinance dataframe and return standard OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    cleaned = df.copy()

    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = cleaned.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(column in cleaned.columns for column in required):
        return pd.DataFrame()

    cleaned = cleaned[required].copy()

    for column in required:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    cleaned = cleaned[
        ~cleaned.index.duplicated(keep="last")
    ].sort_index()

    return cleaned


def download(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Download price data with retries."""
    for attempt in range(1, 4):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
                threads=False,
                prepost=False,
            )

            df = clean(df)

            if not df.empty:
                return df

        except Exception as exc:
            print(
                f"  DATA ERROR {symbol} {interval} "
                f"attempt {attempt}/3: {exc}"
            )

        if attempt < 3:
            time.sleep(attempt)

    return pd.DataFrame()


def completed_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the currently forming intraday candle.

    The scanner should base its calculations on completed candles.
    """
    if df.empty or len(df) < 2:
        return df

    interval = df.index.to_series().diff().dropna().median()

    if pd.isna(interval):
        return df.iloc[:-1].copy()

    try:
        index_timezone = getattr(df.index, "tz", None)

        if index_timezone is not None:
            now = pd.Timestamp.now(tz=index_timezone)
        else:
            now = pd.Timestamp.now()

        completed = df[df.index + interval <= now].copy()

        if completed.empty:
            return df.iloc[:-1].copy()

        return completed

    except Exception:
        return df.iloc[:-1].copy()


# ============================================================
# STATE FUNCTIONS
# ============================================================

def load_state() -> Dict[str, Any]:
    """Load previously sent alerts."""
    try:
        state = json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )

        if not isinstance(state, dict):
            return {"alerts": {}}

        state.setdefault("alerts", {})

        if not isinstance(state["alerts"], dict):
            state["alerts"] = {}

        return state

    except Exception:
        return {"alerts": {}}


def save_state(state: Dict[str, Any]) -> None:
    """Safely save alert state."""
    temporary_file = STATE_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            state,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(STATE_FILE)


def clean_old_state(state: Dict[str, Any]) -> None:
    """Remove old alert records so the state file does not grow forever."""
    alerts = state.setdefault("alerts", {})

    cutoff = (
        pd.Timestamp.now(tz="UTC")
        - pd.Timedelta(days=STATE_RETENTION_DAYS)
    )

    retained: Dict[str, Any] = {}

    for key, value in alerts.items():
        try:
            if not isinstance(value, dict):
                retained[key] = value
                continue

            sent_at = value.get("sent_at")

            if not sent_at:
                retained[key] = value
                continue

            timestamp = pd.Timestamp(sent_at)

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")

            if timestamp >= cutoff:
                retained[key] = value

        except Exception:
            retained[key] = value

    state["alerts"] = retained


# ============================================================
# POC AND RELATIVE STRENGTH
# ============================================================

def volume_profile_poc(df: pd.DataFrame) -> float:
    """
    Estimate the point of control using volume assigned to
    typical-price buckets.
    """
    if df.empty:
        return np.nan

    sample = df.tail(POC_LOOKBACK_BARS).copy()

    if sample.empty:
        return np.nan

    low = float(sample["Low"].min())
    high = float(sample["High"].max())

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        return float(sample["Close"].iloc[-1])

    typical_price = (
        sample["High"]
        + sample["Low"]
        + sample["Close"]
    ) / 3

    edges = np.linspace(
        low,
        high,
        POC_BINS + 1,
    )

    buckets = np.clip(
        np.digitize(
            typical_price.to_numpy(),
            edges,
        ) - 1,
        0,
        POC_BINS - 1,
    )

    volume_by_bucket = np.zeros(
        POC_BINS,
        dtype=float,
    )

    volumes = (
        sample["Volume"]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    for bucket, bar_volume in zip(buckets, volumes):
        volume_by_bucket[int(bucket)] += float(bar_volume)

    poc_bucket = int(np.argmax(volume_by_bucket))

    return float(
        (
            edges[poc_bucket]
            + edges[poc_bucket + 1]
        ) / 2
    )


def relative_strength(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    lookback: int = RS_LOOKBACK_DAYS,
) -> float:
    """
    Return asset performance minus benchmark performance.

    Example:
        Asset return: +10%
        Benchmark return: +4%
        Relative strength: +6%
    """
    if asset.empty or benchmark.empty:
        return np.nan

    if (
        len(asset) <= lookback
        or len(benchmark) <= lookback
    ):
        return np.nan

    asset_close = asset["Close"].dropna()
    benchmark_close = benchmark["Close"].dropna()

    if (
        len(asset_close) <= lookback
        or len(benchmark_close) <= lookback
    ):
        return np.nan

    asset_return = float(
        asset_close.iloc[-1]
        / asset_close.iloc[-lookback - 1]
        - 1
    )

    benchmark_return = float(
        benchmark_close.iloc[-1]
        / benchmark_close.iloc[-lookback - 1]
        - 1
    )

    return asset_return - benchmark_return


def format_percentage(value: float) -> str:
    """Format a decimal return as a percentage."""
    if not np.isfinite(value):
        return "N/A"

    return f"{value * 100:+.2f}%"


# ============================================================
# ENTRY PLAN
# ============================================================

def calculate_trade_plan(
    bars15: pd.DataFrame,
) -> Dict[str, float]:
    """
    Build a trade plan from completed 15-minute candles.

    Entry:
        Highest recent high plus a small buffer.

    Stop:
        Lowest recent low minus a small buffer.

    PT1:
        Entry plus 2R.

    PT2:
        Entry plus 3R.
    """
    required_bars = max(
        ENTRY_LOOKBACK_BARS,
        STOP_LOOKBACK_BARS,
    )

    if bars15.empty or len(bars15) < required_bars:
        return {}

    entry_sample = bars15.tail(
        ENTRY_LOOKBACK_BARS
    )

    stop_sample = bars15.tail(
        STOP_LOOKBACK_BARS
    )

    trigger_high = float(
        entry_sample["High"].max()
    )

    swing_low = float(
        stop_sample["Low"].min()
    )

    entry = trigger_high + ENTRY_BUFFER
    stop = swing_low - STOP_BUFFER
    risk = entry - stop

    if not all(
        np.isfinite(value)
        for value in [entry, stop, risk]
    ):
        return {}

    if entry <= 0 or stop <= 0 or risk <= 0:
        return {}

    risk_pct = risk / entry * 100

    if risk_pct > MAX_RISK_PCT:
        return {}

    current_price = float(
        bars15["Close"].iloc[-1]
    )

    if current_price <= 0:
        return {}

    pt1 = entry + PT1_R_MULTIPLE * risk
    pt2 = entry + PT2_R_MULTIPLE * risk

    distance_pct = (
        entry / current_price - 1
    ) * 100

    return {
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "risk_pct": risk_pct,
        "pt1": pt1,
        "pt2": pt2,
        "current_price": current_price,
        "distance_pct": distance_pct,
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_alert(text: str) -> bool:
    """Send one Telegram message and print the exact API error on failure."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("  TELEGRAM ERROR: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        print(f"  Token loaded: {bool(token)} | Chat ID loaded: {bool(chat_id)}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text[:3900],
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        print(f"  TELEGRAM CONNECTION ERROR: {type(exc).__name__}: {exc}")
        return False

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_response": response.text}

    if response.status_code != 200 or not payload.get("ok", False):
        print(f"  TELEGRAM FAILED: HTTP {response.status_code} | {payload}")
        return False

    print("  TELEGRAM SENT SUCCESSFULLY")
    return True


# ============================================================
# TICKER ANALYSIS
# ============================================================

def analyze(
    ticker: str,
    state: Dict[str, Any],
    spy_daily: pd.DataFrame,
    theme_daily: pd.DataFrame,
    theme_hourly: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze one ticker and return its alert result."""
    result: Dict[str, Any] = {
        "qualified": False,
        "alert": None,
        "reason": "",
        "score": 0,
    }

    daily = download(
        ticker,
        DAILY_PERIOD,
        "1d",
    )

    hourly = completed_intraday(
        download(
            ticker,
            HOURLY_PERIOD,
            "1h",
        )
    )

    bars15 = completed_intraday(
        download(
            ticker,
            INTRADAY_PERIOD,
            "15m",
        )
    )

    if (
        len(daily) < 40
        or len(hourly) < 25
        or len(bars15) < 25
    ):
        result["reason"] = "insufficient data"
        print(f"  SKIP {ticker}: insufficient data")
        return result

    price = float(
        bars15["Close"].iloc[-1]
    )

    average_dollar_volume = float(
        (
            daily["Close"]
            * daily["Volume"].fillna(0)
        )
        .tail(20)
        .mean()
    )

    if not MIN_PRICE <= price <= MAX_PRICE:
        result["reason"] = "price filter"

        print(
            f"  SKIP {ticker}: price ${price:.2f} "
            f"outside ${MIN_PRICE:.2f}-"
            f"${MAX_PRICE:.2f}"
        )

        return result

    if (
        not np.isfinite(average_dollar_volume)
        or average_dollar_volume < MIN_DOLLAR_VOLUME
    ):
        result["reason"] = "liquidity filter"

        print(
            f"  SKIP {ticker}: average dollar volume "
            f"${average_dollar_volume:,.0f} below "
            f"${MIN_DOLLAR_VOLUME:,.0f}"
        )

        return result

    theme = SECTOR_THEME_MAP.get(
        ticker,
        "SPY",
    )

    poc = volume_profile_poc(bars15)

    rs_spy = relative_strength(
        daily,
        spy_daily,
    )

    rs_theme = relative_strength(
        daily,
        theme_daily,
    )

    checks = {
        "Daily bullish trend": bullish_trend(
            daily
        ),
        "1-hour bullish trend": bullish_trend(
            hourly
        ),
        "15m price above EMA 9 & 21":
            price_above_ema9_21(bars15),
        "Price above POC": bool(
            np.isfinite(poc)
            and price > poc
        ),
        "Bullish theme + positive RS": bool(
            not theme_hourly.empty
            and bullish_trend(theme_hourly)
            and np.isfinite(rs_spy)
            and np.isfinite(rs_theme)
            and rs_spy > 0
            and rs_theme > 0
        ),
    }

    score = sum(
        SCORE_WEIGHTS[name]
        for name, passed in checks.items()
        if passed
    )

    result["score"] = score

    status_text = " | ".join(
        (
            f"{name}="
            f"{'PASS' if passed else 'FAIL'}"
            f" "
            f"({SCORE_WEIGHTS[name] if passed else 0}/"
            f"{SCORE_WEIGHTS[name]})"
        )
        for name, passed in checks.items()
    )

    rs_spy_text = format_percentage(
        rs_spy
    )

    rs_theme_text = format_percentage(
        rs_theme
    )

    print(
        f"  SCORE {ticker}: {score}/100 | "
        f"{status_text} | "
        f"RS vs SPY={rs_spy_text} | "
        f"RS vs {theme}={rs_theme_text}"
    )

    if score < ALERT_THRESHOLD:
        result["reason"] = (
            f"score {score} below threshold "
            f"{ALERT_THRESHOLD}"
        )
        return result

    if (
        REQUIRE_DAILY_TREND
        and not checks["Daily bullish trend"]
    ):
        result["reason"] = (
            "daily bullish trend required"
        )

        print(
            f"  NO ALERT {ticker}: "
            "daily bullish trend is required"
        )

        return result

    if (
        REQUIRE_HOURLY_TREND
        and not checks["1-hour bullish trend"]
    ):
        result["reason"] = (
            "1-hour bullish trend required"
        )

        print(
            f"  NO ALERT {ticker}: "
            "1-hour bullish trend is required"
        )

        return result

    result["qualified"] = True

    print(
        f"  QUALIFIED {ticker}: "
        f"{score}/100 passed scoring requirements"
    )

    plan = calculate_trade_plan(
        bars15
    )

    if not plan:
        result["reason"] = (
            "qualified, but no valid entry plan"
        )

        print(
            f"  NO ALERT {ticker}: score qualified, "
            f"but entry-to-stop risk exceeds "
            f"{MAX_RISK_PCT:.2f}% or the plan is invalid"
        )

        return result

    candle_time = pd.Timestamp(
        bars15.index[-1]
    ).isoformat()

    alert_key = (
        f"{ticker}:{score}:{candle_time}"
    )

    if alert_key in state["alerts"]:
        result["reason"] = "duplicate alert"

        print(
            f"  DUPLICATE {ticker}: alert already "
            f"sent for candle {candle_time}"
        )

        return result

    passed_lines = "\n".join(
        (
            f"{'✅' if passed else '❌'} "
            f"{name}: "
            f"{SCORE_WEIGHTS[name] if passed else 0}/"
            f"{SCORE_WEIGHTS[name]}"
        )
        for name, passed in checks.items()
    )

    if score == 100:
        emoji = "🟢"
        label = "FULL CONFIRMATION"

    elif score >= 90:
        emoji = "🟢"
        label = "VERY STRONG SETUP"

    else:
        emoji = "🟡"
        label = "STRONG SETUP"

    message = (
        f"{emoji} {label}: {ticker}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Score: {score}/100\n\n"

        f"Current price: ${price:.2f}\n"
        f"POC: ${poc:.2f}\n"
        f"Theme ETF: {theme}\n\n"

        f"RS vs SPY "
        f"({RS_LOOKBACK_DAYS}D): "
        f"{rs_spy_text}\n"

        f"RS vs {theme} "
        f"({RS_LOOKBACK_DAYS}D): "
        f"{rs_theme_text}\n\n"

        f"ENTRY PLAN\n"
        f"Wait for the trigger. "
        f"Do not enter immediately.\n\n"

        f"Entry trigger: "
        f"${plan['entry']:.2f}\n"

        f"Stop: "
        f"${plan['stop']:.2f}\n"

        f"Risk per share: "
        f"${plan['risk']:.2f}\n"

        f"Risk percentage: "
        f"{plan['risk_pct']:.2f}%\n"

        f"PT1: "
        f"${plan['pt1']:.2f} "
        f"({PT1_R_MULTIPLE:.1f}R)\n"

        f"PT2: "
        f"${plan['pt2']:.2f} "
        f"({PT2_R_MULTIPLE:.1f}R)\n"

        f"Distance to entry: "
        f"{plan['distance_pct']:+.2f}%\n\n"

        f"SCORING\n"
        f"{passed_lines}\n\n"

        f"Rule: Enter only after price trades "
        f"through the entry trigger. Prefer "
        f"confirmation from a completed "
        f"15-minute candle."
    )

    result["alert"] = {
        "ticker": ticker,
        "score": score,
        "key": alert_key,
        "message": message,
        "candle_time": candle_time,
    }

    result["reason"] = "alert ready"

    return result


# ============================================================
# MAIN SCANNER
# ============================================================

def main() -> None:
    """Run one complete scan."""
    tickers = [
        ticker
        for ticker in WHITELIST
        if ticker not in BLACKLIST
    ][:MAX_TICKERS_PER_RUN]

    if not tickers:
        print("No symbols found in WHITELIST.")
        return

    themes = sorted(
        {
            SECTOR_THEME_MAP.get(
                ticker,
                "SPY",
            )
            for ticker in tickers
        }
    )

    print("Downloading benchmark data...")

    spy_daily = download(
        "SPY",
        DAILY_PERIOD,
        "1d",
    )

    if spy_daily.empty:
        print(
            "FATAL ERROR: Unable to download "
            "SPY daily data."
        )
        return

    theme_daily: Dict[str, pd.DataFrame] = {}
    theme_hourly: Dict[str, pd.DataFrame] = {}

    for theme in themes:
        theme_daily[theme] = download(
            theme,
            DAILY_PERIOD,
            "1d",
        )

        theme_hourly[theme] = completed_intraday(
            download(
                theme,
                HOURLY_PERIOD,
                "1h",
            )
        )

    state = load_state()
    clean_old_state(state)
    save_state(state)

    qualified_count = 0
    ready_count = 0
    sent_count = 0
    failed_count = 0
    duplicate_count = 0
    invalid_plan_count = 0
    error_count = 0

    print()
    print(
        f"Scanning {len(tickers)} "
        f"whitelist symbols..."
    )
    print(
        f"Alert threshold: "
        f"{ALERT_THRESHOLD}/100"
    )
    print(
        f"Daily trend required: "
        f"{REQUIRE_DAILY_TREND}"
    )
    print(
        f"1-hour trend required: "
        f"{REQUIRE_HOURLY_TREND}"
    )
    print()

    for index, ticker in enumerate(
        tickers,
        start=1,
    ):
        theme = SECTOR_THEME_MAP.get(
            ticker,
            "SPY",
        )

        print(
            f"[{index}/{len(tickers)}] "
            f"Checking {ticker} ({theme})..."
        )

        try:
            ticker_theme_daily = (
                theme_daily.get(
                    theme,
                    pd.DataFrame(),
                )
            )

            if ticker_theme_daily.empty:
                ticker_theme_daily = spy_daily

            result = analyze(
                ticker=ticker,
                state=state,
                spy_daily=spy_daily,
                theme_daily=ticker_theme_daily,
                theme_hourly=theme_hourly.get(
                    theme,
                    pd.DataFrame(),
                ),
            )

            if result["qualified"]:
                qualified_count += 1

            reason = result.get(
                "reason",
                "",
            )

            if reason == "duplicate alert":
                duplicate_count += 1

            if (
                result["qualified"]
                and reason
                == "qualified, but no valid entry plan"
            ):
                invalid_plan_count += 1

            alert_data = result.get("alert")

            if alert_data:
                ready_count += 1

                print(
                    f"  SENDING TELEGRAM: "
                    f"{alert_data['ticker']} "
                    f"{alert_data['score']}/100"
                )

                sent = send_alert(
                    alert_data["message"]
                )

                if sent:
                    sent_count += 1

                    state["alerts"][
                        alert_data["key"]
                    ] = {
                        "ticker":
                            alert_data["ticker"],
                        "score":
                            alert_data["score"],
                        "candle_time":
                            alert_data["candle_time"],
                        "sent_at":
                            pd.Timestamp.now(
                                tz="UTC"
                            ).isoformat(),
                    }

                    save_state(state)

                    print(
                        f"  TELEGRAM SENT: "
                        f"{alert_data['ticker']}"
                    )

                else:
                    failed_count += 1

                    print(
                        f"  TELEGRAM FAILED: "
                        f"{alert_data['ticker']}"
                    )

        except Exception as exc:
            error_count += 1

            print(
                f"  ERROR {ticker}: "
                f"{type(exc).__name__}: {exc}"
            )

        time.sleep(
            SLEEP_BETWEEN_TICKERS
        )

    print()
    print("=" * 60)
    print("SCAN COMPLETE")
    print(
        f"Symbols scanned: "
        f"{len(tickers)}"
    )
    print(
        f"Score-qualified setups: "
        f"{qualified_count}"
    )
    print(
        f"Alerts ready with valid plan: "
        f"{ready_count}"
    )
    print(
        f"Telegram alerts sent: "
        f"{sent_count}"
    )
    print(
        f"Duplicate alerts skipped: "
        f"{duplicate_count}"
    )
    print(
        f"Invalid or excessive-risk plans: "
        f"{invalid_plan_count}"
    )
    print(
        f"Telegram failures: "
        f"{failed_count}"
    )
    print(
        f"Ticker errors: "
        f"{error_count}"
    )

    summary_message = (
        "📊 30-MINUTE SCANNER UPDATE\n"
        f"Symbols scanned: {len(tickers)}\n"
        f"Qualified setups: {qualified_count}\n"
        f"Valid trade plans: {ready_count}\n"
        f"Setup alerts sent: {sent_count}\n"
        f"Duplicate alerts skipped: {duplicate_count}\n"
        f"Invalid plans: {invalid_plan_count}\n"
        f"Ticker errors: {error_count}"
    )

    # Always send one scan summary, even when no setup qualifies.
    if not send_alert(summary_message):
        print("  TELEGRAM SUMMARY FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()