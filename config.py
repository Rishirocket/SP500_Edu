from __future__ import annotations

# ============================================================
# STOCK UNIVERSE
# ============================================================

WHITELIST = [
    # Index ETFs
    "SPY",
    "QQQ",
    "IWM",

    # Mega-cap technology
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "ORCL",
    "CRM",
    "NFLX",
    "AMD",
    "QCOM",

    # Semiconductors
    "ARM",
    "MU",
    "MRVL",
    "INTC",
    "SMCI",
    "TER",
    "AMKR",

    # Software / cybersecurity
    "PLTR",
    "CRWD",
    "APP",
    "SNOW",
    "RDDT",

    # Financials / fintech
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    "SOFI",
    "HOOD",

    # Healthcare
    "LLY",
    "UNH",
    "ABBV",
    "MRK",
    "JNJ",
    "HIMS",
    "OSCR",
    "NVO",

    # Energy
    "XOM",
    "CVX",
    "NEE",
    "BE",
    "RUN",

    # Aerospace / space
    "RKLB",
    "ASTS",
    "LUNR",
    "ONDS",

    # Quantum / AI
    "IONQ",
    "RGTI",
    "QBTS",
    "QUBT",
    "SOUN",

    # Industrial / consumer / transportation
    "HON",
    "UBER",
    "AAL",
    "JBLU",
    "LOW",
    "DELL",

    # Biotech / medical
    "MRNA",
    "GH",
    "ILMN",

    # Other growth stocks
    "EOSE",
    "NVTS",
    "HYLN",
    "TEM",
]

BLACKLIST: list[str] = []

MAX_TICKERS_PER_RUN = 100


# ============================================================
# SECTOR / THEME ETF MAP
# ============================================================

SECTOR_THEME_MAP = {
    # Index ETFs
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",

    # Technology
    "AAPL": "XLK",
    "MSFT": "XLK",
    "ORCL": "XLK",
    "CRM": "XLK",
    "PLTR": "XLK",
    "APP": "XLK",
    "SNOW": "XLK",
    "DELL": "XLK",

    # Communications / internet
    "META": "XLC",
    "GOOGL": "XLC",
    "NFLX": "XLC",
    "RDDT": "XLC",

    # Consumer discretionary
    "AMZN": "XLY",
    "TSLA": "XLY",
    "LOW": "XLY",
    "UBER": "XLY",

    # Semiconductors
    "NVDA": "SMH",
    "AVGO": "SMH",
    "AMD": "SMH",
    "QCOM": "SMH",
    "ARM": "SMH",
    "MU": "SMH",
    "MRVL": "SMH",
    "INTC": "SMH",
    "SMCI": "SMH",
    "TER": "SMH",
    "AMKR": "SMH",
    "NVTS": "SMH",

    # Cybersecurity
    "CRWD": "CIBR",

    # Financials
    "JPM": "XLF",
    "BAC": "XLF",
    "GS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "SOFI": "XLF",
    "HOOD": "XLF",

    # Healthcare
    "LLY": "XLV",
    "UNH": "XLV",
    "ABBV": "XLV",
    "MRK": "XLV",
    "JNJ": "XLV",
    "HIMS": "XLV",
    "OSCR": "XLV",
    "NVO": "XLV",
    "GH": "XLV",
    "ILMN": "XLV",

    # Biotechnology
    "MRNA": "XBI",
    "TEM": "XBI",

    # Energy
    "XOM": "XLE",
    "CVX": "XLE",

    # Utilities / clean energy
    "NEE": "XLU",
    "BE": "ICLN",
    "RUN": "ICLN",
    "EOSE": "ICLN",

    # Industrials / aerospace
    "HON": "XLI",
    "RKLB": "ARKX",
    "ASTS": "ARKX",
    "LUNR": "ARKX",
    "ONDS": "ARKX",
    "HYLN": "XLI",

    # Airlines
    "AAL": "JETS",
    "JBLU": "JETS",

    # Quantum / artificial intelligence
    "IONQ": "QTUM",
    "RGTI": "QTUM",
    "QBTS": "QTUM",
    "QUBT": "QTUM",
    "SOUN": "BOTZ",
}


# ============================================================
# DATA SETTINGS
# ============================================================

DAILY_PERIOD = "1y"
HOURLY_PERIOD = "3mo"
INTRADAY_PERIOD = "1mo"

SLEEP_BETWEEN_TICKERS = 0.25

DATA_DIR = "scanner_data"


# ============================================================
# PRICE AND LIQUIDITY FILTERS
# ============================================================

MIN_PRICE = 2.00
MAX_PRICE = 1000.00

# Average daily dollar volume over the previous 20 sessions.
MIN_DOLLAR_VOLUME = 20_000_000


# ============================================================
# FIVE-FACTOR WEIGHTED SCORING
# ============================================================

SCORE_WEIGHTS = {
    "Daily bullish trend": 25,
    "1-hour bullish trend": 25,
    "15m price above EMA 9 & 21": 15,
    "Price above POC": 15,
    "Bullish theme + positive RS": 20,
}

# Possible qualified scores with required trends are normally:
# 85, 90 and 100.
ALERT_THRESHOLD = 85

# Higher-timeframe trends must pass even when the score qualifies.
REQUIRE_DAILY_TREND = True
REQUIRE_HOURLY_TREND = True


# ============================================================
# RELATIVE STRENGTH
# ============================================================

# Number of daily trading bars used for relative strength.
RS_LOOKBACK_DAYS = 20


# ============================================================
# VOLUME PROFILE / POC
# ============================================================

POC_LOOKBACK_BARS = 100
POC_BINS = 24


# ============================================================
# ENTRY PLAN
# ============================================================

# Entry is the highest high from the latest completed 15-minute
# candles plus the entry buffer.
ENTRY_LOOKBACK_BARS = 2
ENTRY_BUFFER = 0.01

# Stop is below the lowest low from these completed 15-minute bars.
STOP_LOOKBACK_BARS = 8
STOP_BUFFER = 0.01

# Reject setups whose distance from entry to stop is too wide.
MAX_RISK_PCT = 4.00

PT1_R_MULTIPLE = 2.0
PT2_R_MULTIPLE = 3.0


# ============================================================
# DUPLICATE ALERT CONTROL
# ============================================================

# Delete alert-state records older than this number of days.
STATE_RETENTION_DAYS = 7