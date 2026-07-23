from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from config import *  # noqa: F403
except ImportError:
    pass


# Existing config.py values are honored; environment variables are fallbacks.
DATA_DIR = globals().get("DATA_DIR", "data")
WHITELIST = globals().get("WHITELIST", [])
BLACKLIST = globals().get("BLACKLIST", [])
MAX_TICKERS_PER_RUN = int(globals().get("MAX_TICKERS_PER_RUN", 75))
SLEEP_BETWEEN_TICKERS = float(globals().get("SLEEP_BETWEEN_TICKERS", 0.4))

MIN_PRICE = float(globals().get("MIN_PRICE", 10))
MAX_PRICE = float(globals().get("MAX_PRICE", 1000))
MIN_DOLLAR_VOLUME = float(globals().get("MIN_DOLLAR_VOLUME", 20_000_000))
MIN_SAME_TIME_RVOL = float(globals().get("MIN_SAME_TIME_RVOL", 2.0))
MIN_UP_DOWN_RATIO = float(globals().get("MIN_UP_DOWN_RATIO", 1.2))
LEVEL_TOLERANCE_ATR = float(globals().get("LEVEL_TOLERANCE_ATR", 0.25))
LEVEL_LOOKBACK_BARS = int(globals().get("LEVEL_LOOKBACK_BARS", 160))
MIN_LEVEL_TOUCHES = int(globals().get("MIN_LEVEL_TOUCHES", 2))
BREAKOUT_BUFFER_ATR = float(globals().get("BREAKOUT_BUFFER_ATR", 0.05))
STOP_BUFFER_ATR = float(globals().get("STOP_BUFFER_ATR", 0.10))
MAX_STOP_PCT = float(globals().get("MAX_STOP_PCT", 1.50))
PT1_PCT = float(globals().get("PT1_PCT", 2.0))
PT2_PCT = float(globals().get("PT2_PCT", 3.0))
MIN_RR_TO_PT1 = float(globals().get("MIN_RR_TO_PT1", 2.0))
MIN_DAILY_ATR_PCT = float(globals().get("MIN_DAILY_ATR_PCT", 1.20))
MIN_3DAY_CAPACITY_PCT = float(globals().get("MIN_3DAY_CAPACITY_PCT", 2.00))
TARGET_OBSTACLE_BUFFER_ATR = float(globals().get("TARGET_OBSTACLE_BUFFER_ATR", 0.15))
MIN_PT2_PCT = float(globals().get("MIN_PT2_PCT", 2.25))

# Adaptive 2-year behavior model. These remain configurable from config.py.
HISTORY_PERIOD = str(globals().get("HISTORY_PERIOD", "2y"))
MIN_SETUP_SCORE = float(globals().get("MIN_SETUP_SCORE", 66.0))
MIN_WATCH_SCORE = float(globals().get("MIN_WATCH_SCORE", 54.0))
WATCH_DISTANCE_PCT = float(globals().get("WATCH_DISTANCE_PCT", 2.00))
SEND_WATCH_ALERTS = bool(globals().get("SEND_WATCH_ALERTS", True))
MIN_BEHAVIOR_SAMPLES = int(globals().get("MIN_BEHAVIOR_SAMPLES", 12))
MIN_HISTORICAL_WIN_RATE = float(globals().get("MIN_HISTORICAL_WIN_RATE", 0.52))
BEHAVIOR_FORWARD_DAYS = int(globals().get("BEHAVIOR_FORWARD_DAYS", 3))

Path(DATA_DIR).mkdir(exist_ok=True)
STATE_FILE = Path(DATA_DIR) / "swing_signal_state.json"
UNIVERSE_FILE = Path(DATA_DIR) / "sp500_sectors.csv"
SECTOR_CACHE_FILE = Path(DATA_DIR) / "ticker_sector_cache.json"

SECTOR_ETFS = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["Close"].shift()
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def load_state() -> Dict:
    try:
        state = json.loads(STATE_FILE.read_text())
        state.setdefault("positions", {})
        state.setdefault("notices", {})
        return state
    except Exception:
        return {"positions": {}, "notices": {}}


def load_sector_cache() -> Dict[str, str]:
    try:
        return json.loads(SECTOR_CACHE_FILE.read_text())
    except Exception:
        return {}


def resolve_sector(ticker: str, known: Optional[str], cache: Dict[str, str]) -> Optional[str]:
    if known and not pd.isna(known):
        return str(known)
    if ticker in cache:
        return cache[ticker]
    try:
        sector = yf.Ticker(ticker).get_info().get("sector")
        if sector:
            cache[ticker] = str(sector)
            SECTOR_CACHE_FILE.write_text(json.dumps(cache, indent=2))
            return str(sector)
    except Exception:
        pass
    return None


def save_state(state: Dict) -> None:
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, default=str))
    temp.replace(STATE_FILE)


def safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def update_universe() -> pd.DataFrame:
    today = date.today().isoformat()
    if UNIVERSE_FILE.exists():
        cached = pd.read_csv(UNIVERSE_FILE)
        if not cached.empty and str(cached.iloc[0].get("updated_at", "")) == today:
            return cached

    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        result = pd.DataFrame(
            {
                "ticker": table["Symbol"].map(safe_symbol),
                "sector": table["GICS Sector"],
                "updated_at": today,
            }
        )
        yahoo_sector = {
            "Materials": "Basic Materials",
            "Communication Services": "Communication Services",
            "Consumer Discretionary": "Consumer Cyclical",
            "Consumer Staples": "Consumer Defensive",
            "Energy": "Energy",
            "Financials": "Financial Services",
            "Health Care": "Healthcare",
            "Industrials": "Industrials",
            "Real Estate": "Real Estate",
            "Information Technology": "Technology",
            "Utilities": "Utilities",
        }
        result["sector"] = result["sector"].map(yahoo_sector)
        result.to_csv(UNIVERSE_FILE, index=False)
        return result
    except Exception:
        if UNIVERSE_FILE.exists():
            return pd.read_csv(UNIVERSE_FILE)
        return pd.DataFrame({"ticker": WHITELIST, "sector": None})


def get_run_universe(universe: pd.DataFrame) -> List[str]:
    symbols = list(WHITELIST) if WHITELIST else universe["ticker"].tolist()
    symbols = sorted({safe_symbol(x) for x in symbols if x not in BLACKLIST})
    return symbols[:MAX_TICKERS_PER_RUN]


def completed_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Yahoo's currently forming intraday candle."""
    if df.empty:
        return df
    interval = df.index.to_series().diff().dropna().median()
    if pd.isna(interval):
        return df
    now = pd.Timestamp.now(tz=df.index.tz)
    return df[df.index + interval <= now].copy()


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = ["Open", "High", "Low", "Close", "Volume"]
    return df[columns].dropna(subset=["Open", "High", "Low", "Close"]).copy()


def pivot_points(df: pd.DataFrame, kind: str, wing: int = 2) -> List[Tuple[int, float]]:
    values = df["High"] if kind == "high" else df["Low"]
    points = []
    for i in range(wing, len(df) - wing):
        window = values.iloc[i - wing : i + wing + 1]
        value = float(values.iloc[i])
        if kind == "high" and value == float(window.max()):
            points.append((i, value))
        elif kind == "low" and value == float(window.min()):
            points.append((i, value))
    return points


def structural_level(
    base: pd.DataFrame, direction: str, reference: float
) -> Optional[Dict[str, float]]:
    """Cluster repeatedly tested 1-hour pivots; no fixed-day high/low."""
    base = base.tail(LEVEL_LOOKBACK_BARS).copy()
    if len(base) < 40:
        return None
    current_atr = float(atr(base).iloc[-1])
    if not np.isfinite(current_atr) or current_atr <= 0:
        return None
    kind = "high" if direction == "LONG" else "low"
    pivots = pivot_points(base, kind)
    tolerance = current_atr * LEVEL_TOLERANCE_ATR
    clusters: List[List[Tuple[int, float]]] = []
    for point in pivots:
        placed = False
        for cluster in clusters:
            center = float(np.mean([p[1] for p in cluster]))
            if abs(point[1] - center) <= tolerance:
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])

    candidates = []
    for cluster in clusters:
        if len(cluster) < MIN_LEVEL_TOUCHES:
            continue
        prices = [p[1] for p in cluster]
        center = float(np.mean(prices))
        # Avoid counting consecutive bars from one rejection as separate tests.
        distinct = [cluster[0]]
        for point in cluster[1:]:
            if point[0] - distinct[-1][0] >= 3:
                distinct.append(point)
        if len(distinct) < MIN_LEVEL_TOUCHES:
            continue
        candidates.append(
            {
                "level": center,
                "low": min(prices),
                "high": max(prices),
                "touches": len(distinct),
                "atr": current_atr,
                "first_touch": distinct[0][0],
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item["level"] - reference))


def latest_swings(daily: pd.DataFrame) -> Tuple[List[float], List[float]]:
    highs = [value for _, value in pivot_points(daily.tail(160), "high")]
    lows = [value for _, value in pivot_points(daily.tail(160), "low")]
    return highs, lows


def up_down_volume_ratio(daily: pd.DataFrame, length: int = 20) -> float:
    sample = daily.tail(length)
    up = float(sample.loc[sample["Close"] > sample["Close"].shift(), "Volume"].sum())
    down = float(sample.loc[sample["Close"] < sample["Close"].shift(), "Volume"].sum())
    return up / down if down > 0 else math.inf


def relative_return(asset: pd.DataFrame, benchmark: pd.DataFrame, length: int = 20) -> float:
    if len(asset) <= length or len(benchmark) <= length:
        return np.nan
    asset_return = float(asset["Close"].iloc[-1] / asset["Close"].iloc[-length - 1] - 1)
    benchmark_return = float(benchmark["Close"].iloc[-1] / benchmark["Close"].iloc[-length - 1] - 1)
    return asset_return - benchmark_return


def same_time_rvol(intraday: pd.DataFrame, position: int) -> float:
    candle = intraday.iloc[position]
    timestamp = intraday.index[position]
    earlier = intraday.iloc[:position]
    same_slot = earlier[
        (earlier.index.hour == timestamp.hour)
        & (earlier.index.minute == timestamp.minute)
    ].tail(20)
    average = float(same_slot["Volume"].mean()) if len(same_slot) >= 10 else np.nan
    return float(candle["Volume"] / average) if np.isfinite(average) and average > 0 else np.nan


def close_location(candle: pd.Series) -> float:
    spread = float(candle["High"] - candle["Low"])
    return float((candle["Close"] - candle["Low"]) / spread) if spread > 0 else 0.5


def three_day_capacity_pct(daily: pd.DataFrame, lookback: int = 60) -> float:
    """Median historical three-session high/low range as a percentage."""
    sample = daily.tail(lookback + 3).copy()
    rolling_high = sample["High"].rolling(3).max()
    rolling_low = sample["Low"].rolling(3).min()
    capacity = ((rolling_high - rolling_low) / sample["Close"].shift(3) * 100).dropna()
    return float(capacity.median()) if len(capacity) else np.nan


def clustered_obstacles(
    base: pd.DataFrame, direction: str, entry: float, hourly_atr: float
) -> List[float]:
    """Repeated 1H price barriers lying beyond the proposed entry."""
    base = base.tail(LEVEL_LOOKBACK_BARS).copy()
    kind = "high" if direction == "LONG" else "low"
    tolerance = hourly_atr * LEVEL_TOLERANCE_ATR
    clusters: List[List[Tuple[int, float]]] = []
    for point in pivot_points(base, kind):
        for cluster in clusters:
            center = float(np.mean([p[1] for p in cluster]))
            if abs(point[1] - center) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])

    levels = []
    for cluster in clusters:
        distinct = [cluster[0]]
        for point in cluster[1:]:
            if point[0] - distinct[-1][0] >= 3:
                distinct.append(point)
        if len(distinct) < MIN_LEVEL_TOUCHES:
            continue
        center = float(np.mean([p[1] for p in cluster]))
        if direction == "LONG" and center > entry:
            levels.append(center)
        elif direction == "SHORT" and center < entry:
            levels.append(center)
    return sorted(levels, reverse=(direction == "SHORT"))


def target_analysis(
    direction: str,
    entry: float,
    level: Dict[str, float],
    hourly_base: pd.DataFrame,
    daily: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    daily_atr = float(atr(daily).iloc[-1])
    atr_pct = daily_atr / entry * 100
    capacity_pct = three_day_capacity_pct(daily)
    if (
        not np.isfinite(atr_pct)
        or atr_pct < MIN_DAILY_ATR_PCT
        or not np.isfinite(capacity_pct)
        or capacity_pct < MIN_3DAY_CAPACITY_PCT
    ):
        return None

    # Base height starts at the first repeated test of the breakout boundary.
    structural_sample = hourly_base.tail(LEVEL_LOOKBACK_BARS).copy()
    start = max(0, int(level.get("first_touch", 0)))
    after_first_touch = structural_sample.iloc[start:]
    if after_first_touch.empty:
        return None
    if direction == "LONG":
        opposite_boundary = float(after_first_touch["Low"].min())
        base_height = level["level"] - opposite_boundary
    else:
        opposite_boundary = float(after_first_touch["High"].max())
        base_height = opposite_boundary - level["level"]
    measured_pct = base_height / entry * 100
    if measured_pct < PT1_PCT:
        return None

    obstacles = clustered_obstacles(hourly_base, direction, entry, level["atr"])
    next_obstacle = obstacles[0] if obstacles else np.nan
    if np.isfinite(next_obstacle):
        obstacle_room_pct = abs(next_obstacle - entry) / entry * 100
        if obstacle_room_pct < PT1_PCT:
            return None
        obstacle_buffer_pct = TARGET_OBSTACLE_BUFFER_ATR * level["atr"] / entry * 100
        usable_obstacle_pct = obstacle_room_pct - obstacle_buffer_pct
    else:
        obstacle_room_pct = math.inf
        usable_obstacle_pct = math.inf

    pt2_pct = min(PT2_PCT, measured_pct, usable_obstacle_pct)
    if pt2_pct < MIN_PT2_PCT:
        return None
    if direction == "LONG":
        pt1 = entry * (1 + PT1_PCT / 100)
        pt2 = entry * (1 + pt2_pct / 100)
    else:
        pt1 = entry * (1 - PT1_PCT / 100)
        pt2 = entry * (1 - pt2_pct / 100)
    return {
        "pt1": pt1,
        "pt2": pt2,
        "pt2_pct": pt2_pct,
        "atr_pct": atr_pct,
        "capacity_pct": capacity_pct,
        "measured_pct": measured_pct,
        "next_obstacle": next_obstacle,
        "obstacle_room_pct": obstacle_room_pct,
    }



def percentile_rank(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or not np.isfinite(value):
        return 0.5
    return float((clean <= value).mean())


def behavior_profile(daily: pd.DataFrame, direction: str) -> Dict[str, float]:
    """Learn how this ticker historically behaves after its own momentum breaks.

    A sample is a daily close beyond the prior 20-day extreme with above-normal
    volume. Success means price reaches a volatility-adjusted objective before a
    volatility-adjusted adverse move during the following sessions.
    """
    data = daily.copy()
    data["atr"] = atr(data)
    data["atr_pct"] = data["atr"] / data["Close"] * 100
    data["vol_ratio"] = data["Volume"] / data["Volume"].rolling(20).median()
    data["prior_high"] = data["High"].rolling(20).max().shift(1)
    data["prior_low"] = data["Low"].rolling(20).min().shift(1)
    data["ema20"] = ema(data["Close"], 20)
    data["ema50"] = ema(data["Close"], 50)

    samples = []
    end = len(data) - BEHAVIOR_FORWARD_DAYS
    for i in range(55, max(55, end)):
        row = data.iloc[i]
        if not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue
        if direction == "LONG":
            event = (
                row["Close"] > row["prior_high"]
                and row["Close"] > row["ema20"] > row["ema50"]
                and row["vol_ratio"] >= 1.15
            )
        else:
            event = (
                row["Close"] < row["prior_low"]
                and row["Close"] < row["ema20"] < row["ema50"]
                and row["vol_ratio"] >= 1.15
            )
        if not event:
            continue

        future = data.iloc[i + 1 : i + 1 + BEHAVIOR_FORWARD_DAYS]
        entry = float(row["Close"])
        unit_atr = float(row["atr"])
        if direction == "LONG":
            favorable = float(future["High"].max() - entry)
            adverse = float(entry - future["Low"].min())
        else:
            favorable = float(entry - future["Low"].min())
            adverse = float(future["High"].max() - entry)
        target = max(entry * PT1_PCT / 100, unit_atr * 1.20)
        stop = min(entry * MAX_STOP_PCT / 100, unit_atr * 0.90)
        samples.append(
            {
                "win": float(favorable >= target and adverse < stop),
                "favorable_pct": favorable / entry * 100,
                "adverse_pct": max(0.0, adverse / entry * 100),
                "atr_pct": float(row["atr_pct"]),
                "volume_ratio": float(row["vol_ratio"]),
            }
        )

    if not samples:
        return {
            "samples": 0.0,
            "win_rate": 0.50,
            "median_move_pct": np.nan,
            "median_adverse_pct": np.nan,
            "atr_percentile": 0.50,
        }
    frame = pd.DataFrame(samples)
    current_atr_pct = float(data["atr_pct"].iloc[-1])
    return {
        "samples": float(len(frame)),
        "win_rate": float(frame["win"].mean()),
        "median_move_pct": float(frame["favorable_pct"].median()),
        "median_adverse_pct": float(frame["adverse_pct"].median()),
        "atr_percentile": percentile_rank(data["atr_pct"].tail(252), current_atr_pct),
    }


def setup_score(
    direction: str,
    strength: Dict[str, object],
    behavior: Dict[str, float],
    breakout_rvol: float,
    hold_rvol: float,
    breakout: pd.Series,
    level: Dict[str, float],
) -> float:
    """Score a setup from 0-100 while adapting to each ticker's history."""
    score = 0.0
    win_rate = float(behavior.get("win_rate", 0.50))
    samples = int(behavior.get("samples", 0))
    sample_weight = min(1.0, samples / max(MIN_BEHAVIOR_SAMPLES, 1))
    score += 30.0 * max(0.0, min(1.0, (win_rate - 0.40) / 0.30)) * sample_weight
    score += 8.0 * min(1.0, samples / 30.0)

    if np.isfinite(breakout_rvol):
        score += 18.0 * max(0.0, min(1.0, (breakout_rvol - 1.0) / 2.0))
    if np.isfinite(hold_rvol):
        score += 5.0 * max(0.0, min(1.0, hold_rvol / 1.5))

    location = close_location(breakout)
    score += 10.0 * (location if direction == "LONG" else 1.0 - location)
    score += 7.0 * min(1.0, float(level.get("touches", 0)) / 4.0)

    ratio = float(strength.get("volume_ratio", 1.0))
    rs_spy = float(strength.get("rs_spy", 0.0))
    rs_sector = float(strength.get("rs_sector", 0.0))
    if direction == "LONG":
        score += 8.0 * max(0.0, min(1.0, (ratio - 0.9) / 0.8))
        score += 7.0 * max(0.0, min(1.0, rs_spy / 0.10))
        score += 5.0 * max(0.0, min(1.0, rs_sector / 0.08))
    else:
        score += 8.0 * max(0.0, min(1.0, (1.1 - ratio) / 0.7))
        score += 7.0 * max(0.0, min(1.0, -rs_spy / 0.10))
        score += 5.0 * max(0.0, min(1.0, -rs_sector / 0.08))
    score += 2.0 * float(behavior.get("atr_percentile", 0.5))
    return round(max(0.0, min(100.0, score)), 1)


def strength_snapshot(
    daily: pd.DataFrame, spy: pd.DataFrame, sector: pd.DataFrame
) -> Dict[str, object]:
    close = daily["Close"]
    ema20 = ema(close, 20)
    sma50 = close.rolling(50).mean()
    highs, lows = latest_swings(daily)
    structure_bull = len(highs) >= 2 and len(lows) >= 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]
    structure_bear = len(highs) >= 2 and len(lows) >= 2 and highs[-1] < highs[-2] and lows[-1] < lows[-2]
    ratio = up_down_volume_ratio(daily)
    rs_spy = relative_return(daily, spy)
    rs_sector = relative_return(daily, sector)
    sector_rs = relative_return(sector, spy)
    trend_bull = close.iloc[-1] > ema20.iloc[-1] > sma50.iloc[-1] and ema20.iloc[-1] > ema20.iloc[-5]
    trend_bear = close.iloc[-1] < ema20.iloc[-1] < sma50.iloc[-1] and ema20.iloc[-1] < ema20.iloc[-5]
    bull = structure_bull and ratio >= MIN_UP_DOWN_RATIO and rs_spy > 0 and rs_sector > 0 and sector_rs >= 0 and trend_bull
    bear = structure_bear and ratio <= 1 / MIN_UP_DOWN_RATIO and rs_spy < 0 and rs_sector < 0 and sector_rs <= 0 and trend_bear
    return {
        "bull": bool(bull),
        "bear": bool(bear),
        "volume_ratio": ratio,
        "rs_spy": rs_spy,
        "rs_sector": rs_sector,
        "sector_rs": sector_rs,
    }


def trade_plan(
    direction: str,
    level: Dict[str, float],
    breakout: pd.Series,
    hold: pd.Series,
    hourly_base: pd.DataFrame,
    daily: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    intraday_atr = level["atr"] / math.sqrt(6.5)  # approximate 1H-to-15m scale
    tick = 0.01
    if direction == "LONG":
        entry = float(hold["High"]) + tick
        stop = min(float(hold["Low"]), level["low"]) - STOP_BUFFER_ATR * intraday_atr
    else:
        entry = float(hold["Low"]) - tick
        stop = max(float(hold["High"]), level["high"]) + STOP_BUFFER_ATR * intraday_atr
    targets = target_analysis(direction, entry, level, hourly_base, daily)
    if not targets:
        return None
    pt1, pt2 = targets["pt1"], targets["pt2"]
    risk = abs(entry - stop)
    stop_pct = risk / entry * 100
    reward = abs(pt1 - entry)
    if risk <= 0 or stop_pct > MAX_STOP_PCT or reward / risk < MIN_RR_TO_PT1:
        return None
    return {
        "entry": entry,
        "stop": stop,
        "pt1": pt1,
        "pt2": pt2,
        "risk": risk,
        "rr": reward / risk,
        **targets,
    }


def manage_position(ticker: str, bars: pd.DataFrame, position: Dict) -> List[str]:
    alerts = []
    candle = bars.iloc[-1]
    direction = position["direction"]
    entry, stop = float(position["entry"]), float(position["stop"])
    pt1, pt2 = float(position["pt1"]), float(position["pt2"])
    pt2_pct = float(position.get("pt2_pct", PT2_PCT))
    level = float(position["level"])
    rvol = same_time_rvol(bars, -1)

    if position.get("status") == "ARMED":
        if direction == "LONG" and candle["Low"] <= level:
            position["status"] = "CANCELLED"
            return [f"❌ CANCEL LONG SETUP {ticker}\nPrice lost ${level:.2f} before the ${entry:.2f} entry trigger filled."]
        if direction == "SHORT" and candle["High"] >= level:
            position["status"] = "CANCELLED"
            return [f"❌ CANCEL SHORT SETUP {ticker}\nPrice reclaimed ${level:.2f} before the ${entry:.2f} entry trigger filled."]
        filled = candle["High"] >= entry if direction == "LONG" else candle["Low"] <= entry
        if not filled:
            return alerts
        position["status"] = "OPEN"
        position["filled_at"] = str(bars.index[-1])
        alerts.append(
            f"▶️ ENTRY TRIGGERED {direction} {ticker}\nModel entry ${entry:.2f} traded. Stop ${stop:.2f} | PT1 ${pt1:.2f} | PT2 ${pt2:.2f}"
        )

    if direction == "LONG":
        if candle["Low"] <= stop:
            alerts.append(f"🛑 CLOSE LONG {ticker}\nStop reached: ${stop:.2f}\nClose full model position. Setup invalidated.")
            position["status"] = "CLOSED"
        elif candle["High"] >= pt2:
            alerts.append(f"✅ CLOSE LONG {ticker}\nPT2 reached: ${pt2:.2f} (+{pt2_pct:.2f}%)\nClose remaining model position.")
            position["status"] = "CLOSED"
        elif not position.get("pt1_hit") and candle["High"] >= pt1:
            position["pt1_hit"] = True
            position["stop"] = entry
            alerts.append(f"🎯 PARTIAL CLOSE LONG {ticker}\nPT1 reached: ${pt1:.2f} (+{PT1_PCT:.1f}%)\nClose 50%; stop on remainder moves to ${entry:.2f}.")
        elif candle["Close"] < level and np.isfinite(rvol) and rvol >= MIN_SAME_TIME_RVOL:
            alerts.append(f"⚠️ EARLY CLOSE LONG {ticker}\n15m close ${candle['Close']:.2f} fell below ${level:.2f} on {rvol:.2f}x volume. Breakout failed.")
            position["status"] = "CLOSED"
    else:
        if candle["High"] >= stop:
            alerts.append(f"🛑 CLOSE SHORT {ticker}\nStop reached: ${stop:.2f}\nClose full model position. Setup invalidated.")
            position["status"] = "CLOSED"
        elif candle["Low"] <= pt2:
            alerts.append(f"✅ CLOSE SHORT {ticker}\nPT2 reached: ${pt2:.2f} (-{pt2_pct:.2f}%)\nClose remaining model position.")
            position["status"] = "CLOSED"
        elif not position.get("pt1_hit") and candle["Low"] <= pt1:
            position["pt1_hit"] = True
            position["stop"] = entry
            alerts.append(f"🎯 PARTIAL CLOSE SHORT {ticker}\nPT1 reached: ${pt1:.2f} (-{PT1_PCT:.1f}%)\nClose 50%; stop on remainder moves to ${entry:.2f}.")
        elif candle["Close"] > level and np.isfinite(rvol) and rvol >= MIN_SAME_TIME_RVOL:
            alerts.append(f"⚠️ EARLY CLOSE SHORT {ticker}\n15m close ${candle['Close']:.2f} reclaimed ${level:.2f} on {rvol:.2f}x volume. Breakdown failed.")
            position["status"] = "CLOSED"
    return alerts



def detect_long_patterns(daily: pd.DataFrame, hourly: pd.DataFrame, bars15: pd.DataFrame) -> List[Dict[str, float]]:
    """Return bullish setups found now. Multiple patterns may qualify together."""
    patterns: List[Dict[str, float]] = []
    d = daily.copy()
    h = hourly.copy()
    b = bars15.copy()
    if len(d) < 60 or len(h) < 30 or len(b) < 30:
        return patterns

    close = d["Close"]
    e9, e21, e50 = ema(close, 9), ema(close, 21), ema(close, 50)
    daily_atr = float(atr(d).iloc[-1])
    price = float(b["Close"].iloc[-1])
    if not np.isfinite(daily_atr) or daily_atr <= 0:
        return patterns

    # 1) Flat-top / tight resistance breakout.
    recent = d.iloc[-12:-1]
    resistance = float(recent["High"].max())
    dispersion = float(recent["High"].std() / max(resistance, 1e-9) * 100)
    if dispersion <= 1.25 and price >= resistance * 0.995 and close.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        quality = max(0.0, 18.0 - dispersion * 6.0)
        patterns.append({"name": "Flat-Top Breakout", "quality": quality, "level": resistance})

    # 2) Bull flag: strong impulse followed by shallow, contracting pullback.
    impulse_start = float(d["Close"].iloc[-12])
    impulse_high = float(d["High"].iloc[-6:].max())
    impulse_pct = (impulse_high / impulse_start - 1) * 100 if impulse_start > 0 else 0.0
    pullback_pct = (impulse_high - float(d["Low"].iloc[-5:].min())) / impulse_high * 100
    recent_vol = float(d["Volume"].iloc[-5:].mean())
    prior_vol = float(d["Volume"].iloc[-15:-5].mean())
    if impulse_pct >= 4.0 and pullback_pct <= max(4.5, impulse_pct * 0.55) and price > e21.iloc[-1] and recent_vol <= prior_vol * 1.10:
        level = float(d["High"].iloc[-5:].max())
        quality = min(20.0, 8.0 + impulse_pct * 0.8 - pullback_pct * 0.5)
        patterns.append({"name": "Bull Flag", "quality": quality, "level": level})

    # 3) EMA-21 pullback continuation.
    distance21 = abs(price - float(e21.iloc[-1])) / price * 100
    bounced = float(d["Low"].iloc[-1]) <= float(e21.iloc[-1]) * 1.01 and float(d["Close"].iloc[-1]) > float(e21.iloc[-1])
    if e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1] and e21.iloc[-1] > e21.iloc[-5] and distance21 <= 2.0 and bounced:
        patterns.append({"name": "EMA-21 Pullback", "quality": max(8.0, 18.0 - distance21 * 4.0), "level": float(d["High"].iloc[-1])})

    # 4) Tight consolidation / volatility contraction.
    range5 = (float(d["High"].iloc[-5:].max()) - float(d["Low"].iloc[-5:].min())) / price * 100
    range20 = (float(d["High"].iloc[-20:].max()) - float(d["Low"].iloc[-20:].min())) / price * 100
    if range5 <= min(4.0, range20 * 0.45) and price > e21.iloc[-1] > e50.iloc[-1]:
        patterns.append({"name": "Tight Consolidation", "quality": max(8.0, 19.0 - range5 * 2.5), "level": float(d["High"].iloc[-5:].max())})

    # 5) Momentum continuation on the latest completed 15-minute candle.
    last = b.iloc[-1]
    prev_high = float(b["High"].iloc[-9:-1].max())
    rvol = same_time_rvol(b, -1)
    if float(last["Close"]) > prev_high and close_location(last) >= 0.68 and np.isfinite(rvol) and rvol >= 1.35:
        patterns.append({"name": "Momentum Continuation", "quality": min(22.0, 10.0 + rvol * 4.0), "level": prev_high})

    # 6) Bollinger squeeze expansion.
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    width = (4 * std / mid * 100).replace([np.inf, -np.inf], np.nan)
    width_rank = percentile_rank(width.tail(120), float(width.iloc[-2]))
    expansion = float(close.iloc[-1]) > float((mid + 2 * std).iloc[-1])
    if width_rank <= 0.30 and expansion and price > e21.iloc[-1]:
        patterns.append({"name": "Squeeze Expansion", "quality": 18.0 + (0.30 - width_rank) * 10.0, "level": float(d["High"].iloc[-2])})

    # De-duplicate and keep strongest version of each pattern.
    best: Dict[str, Dict[str, float]] = {}
    for item in patterns:
        if item["name"] not in best or item["quality"] > best[item["name"]]["quality"]:
            best[item["name"]] = item
    return sorted(best.values(), key=lambda x: x["quality"], reverse=True)

def analyze_ticker(
    ticker: str, sector_etf: str, state: Dict, benchmark_cache: Dict[str, pd.DataFrame]
) -> List[str]:
    alerts: List[str] = []
    daily = clean_history(yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False, auto_adjust=True, multi_level_index=False))
    hourly = completed_intraday(clean_history(yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True, multi_level_index=False)))
    bars15 = completed_intraday(clean_history(yf.download(ticker, period="30d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)))
    if len(daily) < 120 or len(hourly) < 50 or len(bars15) < 50:
        return alerts

    price = float(bars15["Close"].iloc[-1])
    dollar_volume = float((daily["Close"] * daily["Volume"]).tail(20).mean())
    if not (MIN_PRICE <= price <= MAX_PRICE) or dollar_volume < MIN_DOLLAR_VOLUME:
        return alerts

    existing = state["positions"].get(ticker)
    if existing and existing.get("status") in {"ARMED", "OPEN"}:
        alerts.extend(manage_position(ticker, bars15, existing))
        return alerts

    spy = benchmark_cache["SPY"]
    sector = benchmark_cache.get(sector_etf, spy)
    strength = strength_snapshot(daily, spy, sector)
    behavior = behavior_profile(daily, "LONG")
    patterns = detect_long_patterns(daily, hourly, bars15)
    if not patterns:
        return alerts

    breakout, hold = bars15.iloc[-2], bars15.iloc[-1]
    breakout_rvol = same_time_rvol(bars15, -2)
    hold_rvol = same_time_rvol(bars15, -1)
    level = structural_level(hourly.iloc[:-2], "LONG", price)
    if not level:
        # Pattern-derived fallback level so good continuations are not discarded.
        chosen_level = float(patterns[0]["level"])
        hourly_atr = float(atr(hourly).iloc[-1])
        level = {"level": chosen_level, "low": chosen_level - 0.15 * hourly_atr, "high": chosen_level, "touches": 1, "atr": hourly_atr, "first_touch": max(0, len(hourly.tail(LEVEL_LOOKBACK_BARS)) - 20)}

    base_score = setup_score("LONG", strength, behavior, breakout_rvol, hold_rvol, breakout, level)
    pattern_points = min(28.0, sum(float(p["quality"]) for p in patterns[:2]))
    trend_bonus = 5.0 if strength["bull"] else 0.0
    score = min(100.0, base_score * 0.78 + pattern_points + trend_bonus)

    # Require price confirmation, but no longer require one identical breakout structure for every stock.
    trigger_level = max(float(p["level"]) for p in patterns[:2])
    confirmed = price >= trigger_level * 0.997 or (float(hold["Close"]) > float(hold["Open"]) and close_location(hold) >= 0.62)
    # Historical behavior improves the score, but it must not eliminate a fresh momentum setup.
    history_ok = behavior["samples"] < 20 or behavior["win_rate"] >= 0.40

    # Accept either intraday relative volume or strong current daily participation.
    avg_daily_volume = float(daily["Volume"].iloc[-21:-1].mean())
    daily_volume_ratio = (
        float(daily["Volume"].iloc[-1]) / avg_daily_volume
        if np.isfinite(avg_daily_volume) and avg_daily_volume > 0
        else 1.0
    )
    volume_ok = (
        (np.isfinite(breakout_rvol) and breakout_rvol >= 1.05)
        or (np.isfinite(hold_rvol) and hold_rvol >= 1.05)
        or daily_volume_ratio >= 1.10
    )

    entry = max(float(hold["High"]) + 0.01, trigger_level + 0.01)
    daily_atr = float(atr(daily).iloc[-1])
    stop = min(float(daily["Low"].iloc[-3:].min()), entry - 0.85 * daily_atr)
    risk = entry - stop
    if risk <= 0:
        return alerts
    stop_pct = risk / entry * 100
    pt1 = entry + max(1.5 * risk, entry * 0.018)
    pt2 = entry + max(2.4 * risk, entry * 0.030)
    rr = (pt1 - entry) / risk

    distance_to_trigger = max(0.0, (entry - price) / price * 100)
    near_trigger = distance_to_trigger <= WATCH_DISTANCE_PCT

    action = None
    if score >= MIN_SETUP_SCORE and confirmed and history_ok and volume_ok and stop_pct <= 4.0:
        action = "BUY"
    elif score >= MIN_WATCH_SCORE and near_trigger and history_ok:
        action = "WATCH"
    if not action:
        print(
            f"  Candidate {ticker}: score={score:.1f}, patterns={patterns[0]['name']}, "
            f"distance={distance_to_trigger:.2f}%, history={behavior['win_rate']*100:.0f}%, "
            f"volume_ok={volume_ok}, confirmed={confirmed}"
        )
        return alerts

    pattern_names = ", ".join(p["name"] for p in patterns[:3])
    notice_key = f"{ticker}:{action}:{date.today()}:{patterns[0]['name']}"
    if notice_key in state["notices"]:
        return alerts
    state["notices"][notice_key] = True

    if action == "BUY":
        alerts.append(
            f"🚀 BUY SETUP: {ticker} | Score {score:.1f}/100\n"
            f"Patterns: {pattern_names}\n"
            f"Entry trigger: ${entry:.2f} | Stop: ${stop:.2f} ({stop_pct:.2f}%)\n"
            f"PT1: ${pt1:.2f} | PT2: ${pt2:.2f} | RR to PT1: {rr:.2f}R\n"
            f"Volume: break {breakout_rvol:.2f}x | latest {hold_rvol:.2f}x | daily {daily_volume_ratio:.2f}x\n"
            f"RS vs SPY: {strength['rs_spy']*100:+.2f}% | RS vs {sector_etf}: {strength['rs_sector']*100:+.2f}%\n"
            f"2-year behavior: {int(behavior['samples'])} samples | Win rate {behavior['win_rate']*100:.1f}% | Median move {behavior['median_move_pct']:.2f}%\n"
            f"Wait for ${entry:.2f} to trade; this is not an automatic market order."
        )
        state["positions"][ticker] = {
            "entry": entry, "stop": stop, "pt1": pt1, "pt2": pt2,
            "pt2_pct": (pt2 / entry - 1) * 100, "direction": "LONG",
            "level": trigger_level, "status": "ARMED", "pt1_hit": False,
            "opened_at": str(bars15.index[-1]), "pattern": pattern_names,
        }
    elif SEND_WATCH_ALERTS:
        alerts.append(
            f"👀 LONG WATCH: {ticker} | Score {score:.1f}/100\n"
            f"Patterns: {pattern_names}\n"
            f"Current: ${price:.2f} | Trigger: ${entry:.2f} | Distance: {distance_to_trigger:.2f}%\n"
            f"Historical win rate: {behavior['win_rate']*100:.1f}% ({int(behavior['samples'])} samples)."
        )
    return alerts


def send_alert(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("\n" + text + "\n")
        return
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True},
        timeout=15,
    )
    response.raise_for_status()


def main() -> None:
    universe = update_universe()
    tickers = get_run_universe(universe)
    sector_lookup = dict(zip(universe["ticker"], universe["sector"]))
    sector_cache = load_sector_cache()
    sector_etfs = {}
    for ticker in tickers:
        sector = resolve_sector(ticker, sector_lookup.get(ticker), sector_cache)
        sector_etfs[ticker] = SECTOR_ETFS.get(sector, "SPY")
    needed = sorted(set(sector_etfs.values()) | {"SPY"})
    benchmark_cache = {
        symbol: clean_history(yf.download(symbol, period=HISTORY_PERIOD, interval="1d", progress=False, auto_adjust=True, multi_level_index=False))
        for symbol in needed
    }
    state = load_state()
    total = 0
    for ticker in tickers:
        try:
            print(f"Checking {ticker} ({sector_etfs[ticker]})...")
            for alert in analyze_ticker(ticker, sector_etfs[ticker], state, benchmark_cache):
                send_alert(alert)
                total += 1
            save_state(state)
        except Exception as exc:
            print(f"{ticker}: {type(exc).__name__}: {exc}")
        time.sleep(SLEEP_BETWEEN_TICKERS)
    print(f"Done. Total alerts: {total}")


if __name__ == "__main__":
    main()