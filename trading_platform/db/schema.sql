-- Platform schema (NEW_PLAN/trading-plan.md §9 "Possible tables" + §25/§26 support).
-- SQLite, deliberately boring.  All timestamps are naive ISO-8601 strings.

CREATE TABLE IF NOT EXISTS securities (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    first_seen  TEXT
);

CREATE TABLE IF NOT EXISTS quotes (
    symbol      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    bid         REAL,
    ask         REAL,
    last        REAL,
    volume      INTEGER,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, timestamp, source)
);

CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,          -- 'daily' | '15min' | ...
    timestamp   TEXT NOT NULL,          -- bar OPEN time (close = timestamp + timeframe)
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      INTEGER,
    source      TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp, source)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol      TEXT NOT NULL,
    as_of       TEXT NOT NULL,
    payload     TEXT NOT NULL,          -- JSON blob (market cap, P/E, FCF, ...)
    source      TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of, source)
);

CREATE TABLE IF NOT EXISTS universe_membership (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    approved         INTEGER NOT NULL,
    approval_date    TEXT NOT NULL,
    screen_version   TEXT NOT NULL,
    reason           TEXT,
    UNIQUE (ticker, approval_date, screen_version)
);

CREATE TABLE IF NOT EXISTS strategies (
    name        TEXT PRIMARY KEY,
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_name    TEXT NOT NULL,
    version          TEXT NOT NULL,
    params_json      TEXT NOT NULL,
    git_commit       TEXT,
    created_at       TEXT NOT NULL,
    backtest_period  TEXT,
    paper_start_date TEXT,
    status           TEXT NOT NULL DEFAULT 'IDEA',  -- promotion pipeline §18
    PRIMARY KEY (strategy_name, version)
);

-- §11 signal envelope.  signal_id is the §25 idempotency key:
--   strategy + ticker + bar_timestamp + action
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     TEXT NOT NULL UNIQUE,
    ticker        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    version       TEXT NOT NULL,
    bar_timestamp TEXT NOT NULL,
    action        TEXT NOT NULL,        -- BUY | SELL | EXIT
    confidence    REAL,
    target_weight REAL,
    quantity      REAL,
    price_hint    REAL,
    status        TEXT NOT NULL DEFAULT 'NEW',  -- NEW|APPROVED|REJECTED|FILLED|SHADOW|IGNORED
    reject_reason TEXT,
    created_at    TEXT NOT NULL
);

-- §21 orders; mode separates backtest / paper / shadow / live books.
CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      TEXT NOT NULL UNIQUE,
    signal_id     TEXT,
    mode          TEXT NOT NULL,
    portfolio     TEXT NOT NULL DEFAULT 'default',
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL,        -- BUY | SELL
    quantity      REAL NOT NULL,
    order_type    TEXT NOT NULL DEFAULT 'MARKET',
    limit_price   REAL,
    expected_price REAL,
    status        TEXT NOT NULL DEFAULT 'NEW',  -- NEW|REVIEWED|SUBMITTED|FILLED|CANCELLED|PENDING_MANUAL|REJECTED
    kill_switch_hold INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    price       REAL NOT NULL,
    quantity    REAL NOT NULL,
    commission  REAL NOT NULL DEFAULT 0,
    slippage    REAL NOT NULL DEFAULT 0,
    UNIQUE (order_id)                   -- one fill per market order (v1)
);

-- Plan §9 names paper_orders / paper_fills / paper_positions explicitly;
-- mode='paper' rows in orders/fills serve as those books.  Positions:
CREATE TABLE IF NOT EXISTS paper_positions (
    portfolio   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    quantity    REAL NOT NULL,
    avg_cost    REAL NOT NULL,
    realized_pl REAL NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (portfolio, ticker)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    portfolio     TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    cash          REAL NOT NULL,
    positions_value REAL NOT NULL,
    equity        REAL NOT NULL,
    realized_pl   REAL NOT NULL,
    unrealized_pl REAL NOT NULL,
    PRIMARY KEY (portfolio, timestamp)
);

CREATE TABLE IF NOT EXISTS live_orders (
    order_id    TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL,          -- INFO | WARNING | CRITICAL
    detail      TEXT,
    signal_id   TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    severity    TEXT NOT NULL,          -- INFORMATION | ACTION | WARNING | CRITICAL (§28)
    channel     TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT,
    signal_id   TEXT
);

CREATE TABLE IF NOT EXISTS system_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp   TEXT NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL,      -- OK | ERROR
    universe_size   INTEGER,
    quotes_requested INTEGER,
    quotes_received INTEGER,
    api_calls       INTEGER,
    errors          INTEGER,
    rate_limited    INTEGER,
    signals         INTEGER,
    risk_approved   INTEGER,
    risk_rejected   INTEGER,
    orders          INTEGER,
    runtime_sec     REAL,
    timings_json    TEXT,
    notes           TEXT
);

-- Plan §32 research discipline: every experiment is recorded, never lost.
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    TEXT NOT NULL UNIQUE,   -- e.g. EXP-0042
    hypothesis       TEXT NOT NULL,
    strategy         TEXT,
    strategy_version TEXT,
    universe_version TEXT,
    params_json      TEXT,
    train_period     TEXT,
    validation_period TEXT,
    results_json     TEXT,
    decision         TEXT,              -- ACCEPTED | REJECTED | INCONCLUSIVE
    reason           TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bars_lookup ON bars (symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy, version, status);
CREATE INDEX IF NOT EXISTS idx_orders_mode ON orders (mode, portfolio, status);
CREATE INDEX IF NOT EXISTS idx_risk_events_ts ON risk_events (timestamp);
