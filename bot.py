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


def analyze_ticker(
    ticker: str, sector_etf: str, state: Dict, benchmark_cache: Dict[str, pd.DataFrame]
) -> List[str]:
    alerts: List[str] = []
    daily = clean_history(yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False))
    hourly = completed_intraday(clean_history(yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True, multi_level_index=False)))
    bars15 = completed_intraday(clean_history(yf.download(ticker, period="30d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)))
    if len(daily) < 100 or len(hourly) < 50 or len(bars15) < 50:
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

    # The second-last completed 15m candle is the break; the last is acceptance.
    breakout, hold = bars15.iloc[-2], bars15.iloc[-1]
    breakout_rvol = same_time_rvol(bars15, -2)
    hold_rvol = same_time_rvol(bars15, -1)
    directions = []
    if strength["bull"]:
        directions.append("LONG")
    if strength["bear"]:
        directions.append("SHORT")

    for direction in directions:
        level = structural_level(hourly.iloc[:-2], direction, float(breakout["Close"]))
        if not level:
            continue
        buffer = BREAKOUT_BUFFER_ATR * level["atr"]
        if direction == "LONG":
            broke = breakout["Close"] > level["high"] + buffer
            accepted = hold["Low"] > level["low"] and hold["Close"] > level["high"]
            strong_close = close_location(breakout) >= 0.75
        else:
            broke = breakout["Close"] < level["low"] - buffer
            accepted = hold["High"] < level["high"] and hold["Close"] < level["low"]
            strong_close = close_location(breakout) <= 0.25
        volume_ok = np.isfinite(breakout_rvol) and breakout_rvol >= MIN_SAME_TIME_RVOL
        if broke and accepted and strong_close and volume_ok:
            plan = trade_plan(direction, level, breakout, hold, hourly.iloc[:-2], daily)
            if not plan:
                continue
            side = "LONG" if direction == "LONG" else "SHORT"
            sign1 = "+" if direction == "LONG" else "-"
            alerts.append(
                f"🚨 ARM {side}: {ticker}\n"
                f"Structural {'resistance' if direction == 'LONG' else 'support'}: ${level['level']:.2f} ({level['touches']} tests)\n"
                f"Place entry trigger at ${plan['entry']:.2f}; no model position until traded.\nStop: ${plan['stop']:.2f}\n"
                f"PT1: ${plan['pt1']:.2f} ({sign1}{PT1_PCT:.1f}%)\nPT2: ${plan['pt2']:.2f} ({sign1}{plan['pt2_pct']:.2f}%)\n"
                f"Target evidence: ATR {plan['atr_pct']:.2f}% | Historical 3-day capacity {plan['capacity_pct']:.2f}% | Base measured move {plan['measured_pct']:.2f}%\n"
                f"Break volume: {breakout_rvol:.2f}x same-time average | Hold volume: {hold_rvol:.2f}x\n"
                f"Up/down volume ratio: {strength['volume_ratio']:.2f} | RS vs SPY: {strength['rs_spy']*100:+.2f}%\n"
                f"RS vs {sector_etf}: {strength['rs_sector']*100:+.2f}% | Reward/risk to PT1: {plan['rr']:.2f}R"
            )
            state["positions"][ticker] = {
                **plan,
                "direction": direction,
                "level": level["level"],
                "status": "ARMED",
                "pt1_hit": False,
                "opened_at": str(bars15.index[-1]),
            }
        else:
            distance = abs(price - level["level"]) / price * 100
            notice_key = f"{ticker}:{direction}:{level['level']:.2f}:{date.today()}"
            if distance <= 1.0 and notice_key not in state["notices"]:
                state["notices"][notice_key] = True
                alerts.append(
                    f"👀 {direction} WATCH: {ticker}\n"
                    f"Structural {'resistance' if direction == 'LONG' else 'support'}: ${level['level']:.2f} ({level['touches']} tests)\n"
                    f"Current price: ${price:.2f} | Distance: {distance:.2f}%\n"
                    f"Volume/structure/RS/sector qualification passed. Waiting for a strong 15m break on {MIN_SAME_TIME_RVOL:.1f}x volume and a second 15m acceptance candle."
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
        symbol: clean_history(yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True, multi_level_index=False))
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