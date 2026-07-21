"""Configuration for the volume-first structural breakout bot."""

WHITELIST = [
    # Mega Cap / AI Leaders
    "MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "GOOG",
    "META", "AVGO", "TSLA",

    # Financials / Payments
    "JPM", "V", "MA", "BAC",
    "AFRM", "SEZL", "XYZ", "UPST", "SOFI",

    # Consumer / Retail
    "WMT", "COST", "HD", "KO",

    # Healthcare / Pharma
    "LLY", "JNJ", "ABBV", "UNH", "OSCR", "HIMS",

    # Technology / Software
    "ORCL", "CRM", "NFLX", "DDOG", "SNOW", "FROG", "TTWO",

    # Semiconductors
    "AMD", "QCOM",

    # Energy
    "XOM", "CVX",

    # Telecom
    "TMUS",

    # AI / Growth / Momentum
    "PLTR", "RGTI", "LUNR", "HOOD", "SHOP", "LMND", "RDDT",

    # Cybersecurity
    "PANW", "CRWD", "FTNT", "NET", "RBRK", "OKTA",

    # Biotech / Genomics
    "TEM", "BEAM", "TWST", "MRNA",

    # Homebuilders
    "DHI", "TOL", "PHM",

    # Crypto / Data Center / Compute
    "HUT", "MARA", "RIOT", "CLSK",

    # Misc Growth
    "GRRR",
]

BLACKLIST = []

# Scan the complete whitelist. The earlier limit of 60 silently omitted names.
MAX_TICKERS_PER_RUN = 100
SLEEP_BETWEEN_TICKERS = 0.5
DATA_DIR = "data"

# Liquidity universe
MIN_PRICE = 10
MAX_PRICE = 1000
MIN_DOLLAR_VOLUME = 20_000_000

# Volume is the primary breakout confirmation.
# 2.0 means the completed 15-minute breakout candle must have twice the
# average volume of the same 15-minute time slot over prior sessions.
MIN_SAME_TIME_RVOL = 2.0

# Daily accumulation/distribution filter.
MIN_UP_DOWN_RATIO = 1.20

# One-hour structural support/resistance detection.
# A valid level needs repeated, separated pivot tests within 0.25 ATR.
LEVEL_LOOKBACK_BARS = 160
MIN_LEVEL_TOUCHES = 2
LEVEL_TOLERANCE_ATR = 0.25

# Entry must close beyond the structural zone by this volatility buffer.
BREAKOUT_BUFFER_ATR = 0.05

# Risk and exits
STOP_BUFFER_ATR = 0.10
MAX_STOP_PCT = 1.00
PT1_PCT = 2.00
PT2_PCT = 3.00
MIN_RR_TO_PT1 = 2.00

# A 2-3% target must be feasible, not merely printed on the alert.
MIN_DAILY_ATR_PCT = 1.20
MIN_3DAY_CAPACITY_PCT = 2.00
TARGET_OBSTACLE_BUFFER_ATR = 0.15
MIN_PT2_PCT = 2.25
