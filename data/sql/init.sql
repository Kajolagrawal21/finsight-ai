-- ============================================================
-- FinSight AI — TimescaleDB Schema
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ─── Raw OHLCV Stock Data ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stock_prices (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(20)     NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    source      VARCHAR(50)     DEFAULT 'yfinance'
);

-- Convert to hypertable (TimescaleDB magic — auto-partitions by time)
SELECT create_hypertable('stock_prices', 'time', if_not_exists => TRUE);

-- Index for fast symbol queries
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol, time DESC);

-- ─── Computed Technical Indicators ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS technical_indicators (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(20)     NOT NULL,
    rsi_14      DOUBLE PRECISION,
    macd        DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist   DOUBLE PRECISION,
    bb_upper    DOUBLE PRECISION,
    bb_middle   DOUBLE PRECISION,
    bb_lower    DOUBLE PRECISION,
    ema_20      DOUBLE PRECISION,
    ema_50      DOUBLE PRECISION,
    sma_200     DOUBLE PRECISION,
    atr_14      DOUBLE PRECISION,
    obv         DOUBLE PRECISION
);

SELECT create_hypertable('technical_indicators', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON technical_indicators (symbol, time DESC);

-- ─── ML Model Predictions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(20)     NOT NULL,
    model_name      VARCHAR(50)     NOT NULL,
    model_version   VARCHAR(20),
    predicted_price DOUBLE PRECISION,
    direction       VARCHAR(10),    -- 'UP' | 'DOWN' | 'NEUTRAL'
    confidence      DOUBLE PRECISION,
    horizon_days    INT             DEFAULT 1
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

-- ─── News & Sentiment ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_articles (
    id              SERIAL          PRIMARY KEY,
    published_at    TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(20),
    title           TEXT            NOT NULL,
    url             TEXT,
    source          VARCHAR(100),
    sentiment_score DOUBLE PRECISION,   -- -1 to +1
    sentiment_label VARCHAR(20),        -- 'positive' | 'negative' | 'neutral'
    embedding_id    VARCHAR(100),       -- Qdrant vector ID
    summary         TEXT
);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles (symbol, published_at DESC);

-- ─── Portfolio Tracking ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio (
    id          SERIAL          PRIMARY KEY,
    symbol      VARCHAR(20)     NOT NULL,
    quantity    DOUBLE PRECISION NOT NULL,
    avg_cost    DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ     DEFAULT NOW(),
    updated_at  TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── MLflow backend tables are auto-created by MLflow server ─────────────

-- ─── Continuous Aggregate: Daily OHLCV summary ───────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)  AS bucket,
    symbol,
    FIRST(open, time)           AS open,
    MAX(high)                   AS high,
    MIN(low)                    AS low,
    LAST(close, time)           AS close,
    SUM(volume)                 AS volume
FROM stock_prices
GROUP BY bucket, symbol
WITH NO DATA;

-- ─── Seed watchlist symbols ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS watchlist (
    symbol      VARCHAR(20)     PRIMARY KEY,
    company     VARCHAR(100),
    sector      VARCHAR(50),
    active      BOOLEAN         DEFAULT TRUE,
    added_at    TIMESTAMPTZ     DEFAULT NOW()
);

INSERT INTO watchlist (symbol, company, sector) VALUES
    ('AAPL',  'Apple Inc.',             'Technology'),
    ('MSFT',  'Microsoft Corp.',        'Technology'),
    ('GOOGL', 'Alphabet Inc.',          'Technology'),
    ('AMZN',  'Amazon.com Inc.',        'Consumer Cyclical'),
    ('NVDA',  'NVIDIA Corp.',           'Technology'),
    ('TSLA',  'Tesla Inc.',             'Consumer Cyclical'),
    ('JPM',   'JPMorgan Chase',         'Financial Services'),
    ('META',  'Meta Platforms',         'Technology'),
    ('NFLX',  'Netflix Inc.',           'Communication Services'),
    ('SPY',   'S&P 500 ETF',            'ETF')
ON CONFLICT (symbol) DO NOTHING;

\echo '✅ FinSight AI schema initialized successfully!'
