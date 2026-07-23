"""Configuration for the adaptive multi-pattern stock scanner.

Telegram credentials should remain in GitHub Secrets / environment variables:
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

DATA_DIR = "data"

# Stocks scanned on every run. Edit this list whenever needed.
WHITELIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "AVGO", "TSLA", "AMD", "QCOM", "MU", "ARM", "SMCI", "ORCL",
    "CRM", "NFLX", "PLTR", "APP", "CRWD", "DELL", "INTC", "MRVL",
    "JPM", "BAC", "GS", "V", "MA", "SOFI", "HOOD", "COIN",
    "LLY", "UNH", "ABBV", "MRK", "JNJ", "HIMS", "OSCR", "NVO",
    "RKLB", "ASTS", "LUNR", "IONQ", "RGTI", "QBTS", "QUBT", "SOUN",
    "ONDS", "TEM", "EOSE", "BE", "RUN", "UBER", "AAL", "JBLU",
    "XOM", "CVX", "NEE", "GE", "HON", "WMT", "COST", "TGT",
    "RDDT", "SNOW", "SHOP", "PANW", "MSTR", "MARA", "RIOT",
]
BLACKLIST = []

# Runtime controls
MAX_TICKERS_PER_RUN = 75
SLEEP_BETWEEN_TICKERS = 0.20

# Liquidity and price filters
MIN_PRICE = 3.00
MAX_PRICE = 1500.00
MIN_DOLLAR_VOLUME = 8_000_000

# Breakout and volume requirements. These are intentionally moderate so the
# scanner ranks candidates rather than rejecting nearly everything.
MIN_SAME_TIME_RVOL = 1.20
MIN_UP_DOWN_RATIO = 1.05
LEVEL_TOLERANCE_ATR = 0.35
LEVEL_LOOKBACK_BARS = 160
MIN_LEVEL_TOUCHES = 2
BREAKOUT_BUFFER_ATR = 0.00

# Trade-plan controls
STOP_BUFFER_ATR = 0.10
MAX_STOP_PCT = 4.00
PT1_PCT = 1.50
PT2_PCT = 3.00
MIN_RR_TO_PT1 = 1.20
MIN_DAILY_ATR_PCT = 0.70
MIN_3DAY_CAPACITY_PCT = 1.25
TARGET_OBSTACLE_BUFFER_ATR = 0.10
MIN_PT2_PCT = 1.75

# Adaptive two-year behavior model
HISTORY_PERIOD = "2y"
MIN_SETUP_SCORE = 66.0
MIN_WATCH_SCORE = 54.0
WATCH_DISTANCE_PCT = 2.00
SEND_WATCH_ALERTS = True
MIN_BEHAVIOR_SAMPLES = 8
MIN_HISTORICAL_WIN_RATE = 0.45
BEHAVIOR_FORWARD_DAYS = 3